#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Layer
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import backend as K


def read_vocabulary(args):
    # read vocabulary
    raw_v = []
    n_total = 0
    stats = {}
    with open(args.vocab_path, "r") as f:
        for line in f:
            tokens = line.strip().split("\t")
            if len(tokens) < 2:
                continue
            try:
                n = int(tokens[1])
            except ValueError:
                continue
            if len(tokens[0]) < 3:
                continue
            raw_v.append([tokens[0], n])
            n_total += n
            prefix = tokens[0][:3]
            if prefix not in stats:
                stats[prefix] = []
            stats[prefix].append(n)
    # calculate median frequency for each prefix
    med = {k: np.median(v) for k, v in stats.items()}
    # calculate scores
    scores = [[tag, n, n / med[tag[:3]]] for tag, n in raw_v]
    # sort vocabulary by score in the descending order
    s = sorted(scores, key=lambda x: x[2], reverse=True)
    # take only top max_vocab items and make a dictionary
    N = min(len(s), args.max_num_tags)
    voc = [s[i][0] for i in range(N)]
    frq = [s[i][1] for i in range(N)]
    frq = np.array(frq, dtype=np.float32)
    # find n_oov (frequency of out-of-vocabulary tags)
    n_oov = n_total - np.sum(frq)
    # find n_max
    n_max = np.max(frq)
    # convert frq to idf
    idf = np.log(1 + n_max / (1 + frq))
    # calculate idf value for oov tags, (= min idf)
    # notice n_oov is not used here! (intentional)
    idf_oov = np.log(1 + n_max / (1 + n_max))
    # calculate sampling frequency
    # notice that 0th slot is reserved for oov tags
    frq_with_oov = np.zeros(args.max_num_tags + 1, dtype=np.float32)
    frq_with_oov[0] = n_oov
    frq_with_oov[1:N + 1] = frq 
    z = frq_with_oov / n_total
    n_total_neg = np.sum(np.power(frq_with_oov, 0.75))
    frq_p = (1 + np.sqrt(z / args.alpha)) * (args.alpha / z)
    frq_n = np.power(frq_with_oov, 0.75) / n_total_neg
    # return vocabulary and stats
    return voc, idf, idf_oov, frq_p, frq_n


def fasttext_dataset(data_prefix, batch_size=512, shuffle=True):
    # specify a list of files
    dataset = tf.data.Dataset.list_files(data_prefix + "*")
    # read one line at a time from each file & interleave them
    dataset = dataset.interleave(
        lambda x: tf.data.TextLineDataset(x),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # shuffle lines
    if shuffle:
        shuffle_buffer_size = 10000
        dataset = dataset.shuffle(shuffle_buffer_size)
    # explicitly specify each data point is visited only once
    dataset = dataset.repeat(1)
    # apply split so that the ADID is put aside
    dataset = dataset.map(
        lambda x: tf.strings.split(x, maxsplit=1),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # apply batch size & ask tf to prefetch a few batches in advance
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


class PosSample(Layer):
    def __init__(self, frq_p, **kwargs):
        super().__init__(**kwargs)
        self.frq_p = frq_p

    def call(self, tfid_vec_raw):
        eps = tf.random.uniform(tf.shape(tfid_vec_raw))
        tfid_vec = K.in_train_phase(
            tf.where(eps < self.frq_p, tfid_vec_raw, 0),
            tfid_vec_raw)
        return tfid_vec


class NegMask(Layer):
    def __init__(self, frq_n, idf_oov, neg_sampling, **kwargs):
        super().__init__(**kwargs)
        # expected num of samples
        exp_n_samples = len(frq_n) * neg_sampling
        # frq_n is scaled so that it spawns specified num of samples
        self.frq_n = exp_n_samples * frq_n
        self.idf_oov = idf_oov

    def call(self, tfid_vec):
        positive = (tfid_vec >= self.idf_oov)
        eps = tf.random.uniform(tf.shape(tfid_vec))
        negative = (eps < self.frq_n)
        return tf.logical_or(positive, negative)


def compute_kernel(x, y):
    shape_x = tf.shape(x)
    shape_y = tf.shape(y)
    row_x = shape_x[0]
    row_y = shape_y[0]
    col = shape_x[1]
    tiled_x = tf.tile(tf.reshape(x, [row_x, 1, col]), [1, row_y, 1])
    tiled_y = tf.tile(tf.reshape(y, [1, row_y, col]), [row_x, 1, 1])
    mse = tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=-1)
    return tf.exp(-mse / tf.cast(col, tf.float32))


def MMD(z):
    z_true = tf.random.normal(tf.shape(z))
    x_kernel = tf.reduce_mean(compute_kernel(z_true, z_true))
    y_kernel = tf.reduce_mean(compute_kernel(z, z))
    xy_kernel = tf.reduce_mean(compute_kernel(z_true, z))
    return x_kernel + y_kernel - 2 * xy_kernel


def define_model(args):
    # read vocabulary
    voc, idf, idf_oov, frq_p, frq_n = read_vocabulary(args)
    # initialize vectorization layer
    vectorize_layer = TextVectorization(
        max_tokens=args.max_num_tags + 1,
        standardize=None,
        split="whitespace",
        ngrams=None,
        output_mode="tf-idf",
        pad_to_max_tokens=True)
    vectorize_layer.set_vocabulary(voc, idf, oov_df_value=idf_oov)
    # encoder
    text_input = Input(shape=(2,), dtype=tf.string)
    adid, sentence = tf.split(text_input, [1, 1], axis=-1)
    tfid_vec_raw = vectorize_layer(sentence)
    tfid_vec = PosSample(frq_p)(tfid_vec_raw)
    tfid_vec_drop = Dropout(args.dropout)(tfid_vec)
    z = Dense(args.dim_emb)(tfid_vec_drop)
    enc = Model(text_input, [z, adid])
    # decoder
    decoder_input = Input(shape=(args.dim_emb,))
    outputs = Dense(args.max_num_tags + 1)(decoder_input)
    dec = Model(decoder_input, outputs)
    # connect encoder & decoder
    recon = dec(enc(text_input)[0])
    vae = Model(text_input, recon)
    # apply negative sampling mask
    mask = NegMask(frq_n, idf_oov, args.neg_sampling)(tfid_vec)
    tfid_vec_m = tf.boolean_mask(tfid_vec, mask)
    recon_m = tf.boolean_mask(recon, mask)
    # add loss
    recon_loss = tf.reduce_mean(tf.square(tfid_vec_m - recon_m))
    mmd_loss = args.mmd_reg_param * MMD(z)
    vae.add_loss(recon_loss + mmd_loss)
    # compile model
    vae.compile(optimizer="rmsprop")
    return vae, enc


def fmt(x):
    return " ".join([f"{k}:{v:.6f}" for k, v in enumerate(x)])


def calculate_embeddings(enc, args):
    dataset = fasttext_dataset(args.data_prefix, shuffle=False)
    with open(args.embed_path, "w") as f_luf:
        for item in dataset:
            z, adid = enc.predict_on_batch(item)
            z = z.numpy()
            adid = adid.numpy()
            for i in range(z.shape[0]):
                adid_i = adid[i, 0].decode("utf-8")
                print(f"{adid_i}\t{fmt(z[i, :])}", file=f_luf)


def train(args):
    # set up logger
    fmt = "%(levelname)s: %(asctime)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    # define model
    logging.info("defining model")
    vae, enc = define_model(args)
    # train model
    logging.info("starting to train model")
    csv_logger = CSVLogger(args.loss_log)
    vae.fit(
        fasttext_dataset(args.data_prefix),
        epochs=args.epochs, callbacks=[csv_logger])
    logging.info("finished training model")
    # calculate user embeddings (prediction)
    logging.info("calculating user embeddings")
    calculate_embeddings(enc, args)
    # done
    logging.info("done")


if __name__ == "__main__":
    # command-line arguments
    parser = argparse.ArgumentParser("python train_VAE.py")
    parser.add_argument("data_prefix", type=str)
    parser.add_argument("vocab_path", type=str)
    parser.add_argument("embed_path", type=str)
    parser.add_argument("--max_num_tags", type=int, default=20000)
    parser.add_argument("--dim_emb", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--mmd_reg_param", type=float, default=100)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--neg_sampling", type=float, default=0.05)
    parser.add_argument("--loss_log", type=str, default="loss.csv")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    # launch training
    train(args)

