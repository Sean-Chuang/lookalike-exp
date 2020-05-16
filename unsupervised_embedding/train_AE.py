#!/usr/bin/env python3.6

import sys
sys.dont_write_bytecode = True

import datetime

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import torch
from torch import nn, Tensor, FloatTensor
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler


def log(msg):
    time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print('%s %s' % (time, msg))


def tokenizer(x):
    return x.split()


def load_training_data(data, min_df=500):
    values = []
    ad_ids = []
    for line in open(data):
        i = line.index(' ')
        ad_ids.append(line[0:i])
        values.append(line[i + 1:])

    vectorizer = CountVectorizer(binary=True, tokenizer=tokenizer, min_df=min_df)
    X = vectorizer.fit_transform(values).astype(np.uint8)
    feature_names = vectorizer.get_feature_names()

    y_indices = []
    for i, f in enumerate(feature_names):
        prefix = f.split(':', 1)[0]
        if prefix not in ['pixel', 'pv']:
            y_indices.append(i)
    Y = X[:, y_indices]

    return X, Y, ad_ids


def weights_init(m):
    if isinstance(m, nn.Linear):
        w = m.weight.data
        nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))


class Noise(nn.Module):
    # drop training input randomly. `p` is drop rate.
    def __init__(self, p=0.5):
        super().__init__()
        self.noise = torch.Tensor().cuda()
        self.p = p

    def forward(self, x):
        if self.p > 0 and self.training:
            self.noise.resize_as_(x)
            self.noise.bernoulli_(1 - self.p)
            return x.mul_(self.noise)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, n, m):
        super(AutoEncoder, self).__init__()
        self.noise = Noise()
        self.encoder = nn.Sequential(
            nn.Linear(n, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.encoder.apply(weights_init)
        self.decoder = nn.Linear(128, m)

    def forward(self, x):
        x = self.noise(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def to_variable(X):
    X = X.todense().astype(np.float32)
    tensor = torch.FloatTensor(X).cuda()
    variable = Variable(tensor, requires_grad=False)
    return variable


def train(X, Y, max_epoch, batch_size):
    x_size, m = X.shape
    y_size, n = Y.shape

    autoencoder = AutoEncoder(m, n).cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        autoencoder.parameters(),
        lr=0.001,
        betas=(0.9, 0.999)
    )

    log('start train')
    autoencoder.train()
    for epoch in range(max_epoch):
        total_loss = 0
        indices = list(range(x_size))
        batch_sampler = BatchSampler(RandomSampler(indices), batch_size=batch_size, drop_last=False)
        for batch_indices in batch_sampler:
            x_batch = to_variable(X[batch_indices, :])
            y_batch = to_variable(Y[batch_indices, :])
            y_pred = autoencoder(x_batch)
            loss = criterion(y_pred, y_batch)
            total_loss = total_loss + loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        log('epoch=%d, loss=%6.6f' % (epoch, total_loss))
    autoencoder.eval()
    log('end train')

    return autoencoder


def save_feature(autoencoder, outdir, X, ad_ids, batch_size=1000):
    x_size, m = X.shape
    outfile =  open(outdir + 'luf.vec', 'w')
    indices = list(range(x_size))

    features = []
    batch_sampler = BatchSampler(SequentialSampler(indices), batch_size=batch_size, drop_last=False)
    for batch_indices in batch_sampler:
        x_batch = to_variable(X[batch_indices, :])
        v = autoencoder.encoder(x_batch).data.cpu().numpy()
        features.append(v)
    features = np.vstack(features)

    for i in indices:
        f = features[i, :]
        fs = ' '.join(['%d:%.6f' % x for x in enumerate(f)])
        outfile.write(ad_ids[i] + '\t' + fs + '\n')


def main(argv):
    data = argv[0]
    outdir = argv[1]
    max_epoch = int(argv[2])
    batch_size = int(argv[3])
    print(data, outdir, max_epoch, batch_size)

    X, Y, ad_ids = load_training_data(data)
    print('X.shape', X.shape)
    print('Y.shape', Y.shape)

    model = train(X, Y, max_epoch, batch_size)
    save_feature(model, outdir, X, ad_ids)


if __name__ == '__main__':
    main(sys.argv[1:])
