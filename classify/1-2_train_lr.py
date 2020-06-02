#!/usr/bin/env python3
import os, sys
from tqdm import tqdm
import argparse
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# Read user embedding
def get_user_vector(user_emb):
    v_data = {}

    with open(user_emb) as in_f:
        for line in in_f:
            tmp = line.strip().split("\t")
            uid = tmp[0]
            vector = [float(x.split(":")[1]) for x in tmp[1].split(" ")]
            v_data[uid] = vector

    return v_data


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
    for cid in tqdm(cv):
        # Construct training data
        pos = list(set(cv[cid]))
        neg = list(set(vimp[cid]))
        X = np.empty((len(pos) + len(neg), dim))
        y = np.zeros(len(pos) + len(neg))
        y[:len(pos)] = 1.0
        for i, a in enumerate(pos):
            X[i, :] = luf[a]
        for i, a in enumerate(neg):
            X[len(pos) + i, :] = luf[a]
        # Set up parameters
        # To follow the convention in spark.ml, C = 1 / (n * lambda)
        lr_param = {"C": 1.0 / (0.3 * len(y)), "class_weight": "balanced"}
        # Construct pipelines
        pipe_f = Pipeline([('scal', StandardScaler()),
                           ('lr', LogisticRegression(**lr_param))])
        pipe_d = Pipeline([('scal', StandardScaler()),
                           ('lr', LogisticRegression(**lr_param))])
        # Train them
        pipe_f.fit(X, y)
        # Store them
        models[cid] = pipe_f
    return models


# Apply fitted models to test data
def test(models, vimp, cv, luf, result):
    dim = 128
    with open(os.path.join('result', result), "w") as output_file:
        for cid in tqdm(cv):
            # Construct training data
            pos = list(set(cv[cid]))
            neg = list(set(vimp[cid]))
            X = np.empty((len(pos) + len(neg), dim))
            y_true = np.zeros(len(pos) + len(neg))
            y_true[:len(pos)] = 1.0
            for i, a in enumerate(pos):
                X[i, :] = luf[a]
            for i, a in enumerate(neg):
                X[len(pos) + i, :] = luf[a]
            # Load trained models
            pipe_f = models[cid]
            z_f = pipe_f.predict(X)
            z_f_proba = pipe_f.predict_proba(X)[:, 1]
            # Count results
            result_f = [np.count_nonzero(y_true * z_f),
                        np.count_nonzero((1.0 - y_true) * (1.0 - z_f)),
                        np.count_nonzero((1.0 - y_true) * z_f),
                        np.count_nonzero(y_true * (1.0 - z_f))]
            # Calculate F1
            f1_f = f1_score(y_true, z_f)
            # Calculate AP
            sc_f = pipe_f.decision_function(X)
            ap_f = average_precision_score(y_true, sc_f)
            r = sorted(zip(y_true, z_f_proba), key=lambda x: x[1], reverse=True)
            r_res, _ = zip(*r)
            rank_ap_f = average_precision(r_res)
            # Print result
            s_f = "\t".join([str(r) for r in result_f])
            print(f"{cid}\t{s_f}\t{f1_f:.6f}\t{ap_f:.6f}\t{rank_ap_f:.6f}",
                  file=output_file)


# Time function
def get_t():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ranking map
def average_precision(r, step=False):
    r = np.asarray(r) != 0
    if step:
        out = [precision_at_k(r, int(r.size * k)) for k in np.arange(0.1, 1.1, 0.1)]
    else:
        out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("python3 1-2_train_lr.py")
    parser.add_argument("input_data", type=str, help="Input data")
    parser.add_argument("result", type=str, help="Result")
    parser.add_argument("emb", type=str, help="User emb.")
    os.makedirs('result', exist_ok=True)
    args = parser.parse_args()
    print(f"[{get_t()}] reading embdding")
    luf = get_user_vector(args.emb)
    print(f"[{get_t()}] reading data")
    vimp1, vimp2, cv1, cv2 = read_data(args.input_data, set(luf.keys()))
    print(f"[{get_t()}] training")
    models = train(vimp1, cv1, luf)
    print(f"[{get_t()}] test")
    test(models, vimp2, cv2, luf, args.result)
    print(f"[{get_t()}] done")

