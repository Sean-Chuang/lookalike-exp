#!/usr/bin/env python3
import os, sys
import argparse
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# Read user embedding
def get_user_vector(user_emb_v1, user_emb_v2):
    luf = {}
    v1_data, v2_data = {}, {}

    with open(user_emb_v1) as in_f:
        for line in in_f:
            tmp = line.strip().split("\t")
            uid = tmp[0]
            vector = [float(x.split(":")[1]) for x in tmp[1].split(" ")]
            v1_data[uid] = vector

    with open(user_emb_v2) as in_f:
        for line in in_f:
            tmp = line.strip().split("\t")
            uid = tmp[0]
            vector = [float(x.split(":")[1]) for x in tmp[1].split(" ")]
            v2_data[uid] = vector

    for uid in v1_data:
        luf[uid] = [v1_data[uid], v2_data[uid]]

    return luf


# Read data
def read_data(input_data, user_ids):
    vimp1_raw = {}
    vimp2_raw = {}
    cv1_raw = {}
    cv2_raw = {}
    for line in open(input_data):
        tokens = line.strip().split("\t")
        if len(tokens) < 6:
            continue
        cid = tokens[0]
        ad_id = tokens[1]
        if ad_id not in user_ids:
            continue

        vimp1_flag, vimp2_flag = int(tokens[2]), int(tokens[3])
        cv1_flag, cv2_flag = int(tokens[4]), int(tokens[5])

        if vimp1_flag > 0:
            if cid not in vimp1_raw:
                vimp1_raw[cid] = []
            vimp1_raw[cid].append(ad_id)
        elif vimp2_flag > 0:
            if cid not in vimp2_raw:
                vimp2_raw[cid] = []
            vimp2_raw[cid].append(ad_id)
        elif cv1_flag > 0:
            if cid not in cv1_raw:
                cv1_raw[cid] = []
            cv1_raw[cid].append(ad_id)
        elif cv2_flag > 0:
            if cid not in cv2_raw:
                cv2_raw[cid] = []
            cv2_raw[cid].append(ad_id)
        else:
            raise Exception("data error")
    # Just in case
    cids = set(vimp1_raw) & set(vimp2_raw) & set(cv1_raw) & set(cv2_raw)
    vimp1 = {}
    vimp2 = {}
    cv1 = {}
    cv2 = {}
    for cid in cids:
        vimp1[cid] = vimp1_raw[cid]
        vimp2[cid] = vimp2_raw[cid]
        cv1[cid] = cv1_raw[cid]
        cv2[cid] = cv2_raw[cid]
    return vimp1, vimp2, cv1, cv2

                
# Train LR models
def train(vimp, cv, luf):
    dim = 128
    models = {}
    for cid in cv:
        # Construct training data
        pos = list(set(cv[cid]))
        neg = list(set(vimp[cid]))
        Xf = np.empty((len(pos) + len(neg), dim))
        Xd = np.empty((len(pos) + len(neg), dim))
        y = np.zeros(len(pos) + len(neg))
        y[:len(pos)] = 1.0
        for i, a in enumerate(pos):
            Xf[i, :] = luf[a][0]
            Xd[i, :] = luf[a][1]
        for i, a in enumerate(neg):
            Xf[len(pos) + i, :] = luf[a][0]
            Xd[len(pos) + i, :] = luf[a][1]
        # Set up parameters
        # To follow the convention in spark.ml, C = 1 / (n * lambda)
        lr_param = {"C": 1.0 / (0.3 * len(y)), "class_weight": "balanced"}
        # Construct pipelines
        pipe_f = Pipeline([('scal', StandardScaler()),
                           ('lr', LogisticRegression(**lr_param))])
        pipe_d = Pipeline([('scal', StandardScaler()),
                           ('lr', LogisticRegression(**lr_param))])
        # Train them
        pipe_f.fit(Xf, y)
        pipe_d.fit(Xd, y)
        # Store them
        models[cid] = [pipe_f, pipe_d]
    return models


# Apply fitted models to test data
def test(models, vimp, cv, luf, result):
    dim = 128
    with open(os.path.join('result', result), "w") as output_file:
        for cid in cv:
            # Construct training data
            pos = list(set(cv[cid]))
            neg = list(set(vimp[cid]))
            Xf = np.empty((len(pos) + len(neg), dim))
            Xd = np.empty((len(pos) + len(neg), dim))
            y_true = np.zeros(len(pos) + len(neg))
            y_true[:len(pos)] = 1.0
            for i, a in enumerate(pos):
                Xf[i, :] = luf[a][0]
                Xd[i, :] = luf[a][1]
            for i, a in enumerate(neg):
                Xf[len(pos) + i, :] = luf[a][0]
                Xd[len(pos) + i, :] = luf[a][1]
            # Load trained models
            pipe_f, pipe_d = models[cid]
            z_f = pipe_f.predict(Xf)
            z_d = pipe_d.predict(Xd)
            # Count results
            result_f = [np.count_nonzero(y_true * z_f),
                        np.count_nonzero((1.0 - y_true) * (1.0 - z_f)),
                        np.count_nonzero((1.0 - y_true) * z_f),
                        np.count_nonzero(y_true * (1.0 - z_f))]
            result_d = [np.count_nonzero(y_true * z_d),
                        np.count_nonzero((1.0 - y_true) * (1.0 - z_d)),
                        np.count_nonzero((1.0 - y_true) * z_d),
                        np.count_nonzero(y_true * (1.0 - z_d))]
            # Calculate F1
            f1_f = f1_score(y_true, z_f)
            f1_d = f1_score(y_true, z_d)
            # Calculate AP
            sc_f = pipe_f.decision_function(Xf)
            sc_d = pipe_d.decision_function(Xd)
            ap_f = average_precision_score(y_true, sc_f)
            ap_d = average_precision_score(y_true, sc_d)
            # Print result
            s_f = "\t".join([str(r) for r in result_f])
            s_d = "\t".join([str(r) for r in result_d])
            print(f"{cid}\t{s_f}\t{s_d}\t{f1_f:.6f}\t" +
                  f"{f1_d:.6f}\t{ap_f:.6f}\t{ap_d:.6f}",
                  file=output_file)


# Time function
def get_t():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


if __name__ == "__main__":
    data_dir = os.path.join(CURRENT_PATH, 'train_data')
    parser = argparse.ArgumentParser("python3 train_lr.py")
    parser.add_argument("input_data", type=str, help="Input data")
    parser.add_argument("result", type=str, help="Result")
    parser.add_argument("--emb_v1", type=str, default=os.path.join(data_dir, "emb_v1.vec"), help="User emb_v1")
    parser.add_argument("--emb_v2", type=str, default=os.path.join(data_dir, "emb_v2.vec"), help="User emb_v2")
    args = parser.parse_args()
    print(f"[{get_t()}] reading embdding")
    luf = get_user_vector(args.emb_v1, args.emb_v2)
    print(f"[{get_t()}] reading data")
    vimp1, vimp2, cv1, cv2 = read_data(args.input_data, set(luf.keys()))
    print(f"[{get_t()}] training")
    models = train(vimp1, cv1, luf)
    print(f"[{get_t()}] test")
    test(models, vimp2, cv2, luf, args.result)
    print(f"[{get_t()}] done")

