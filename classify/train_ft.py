#!/usr/bin/env python3
import os, sys
import glob
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm 
import random

import fasttext as ft
'''
Input : 
- input data
- user history events
- events embedding
'''

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# Read user embedding
def get_user_events(user_events_file):
    user_events = {}
    file_list = glob.glob(user_events_file + '*')
    for file_name in tqdm(file_list):
        with open(file_name, 'r') as in_f:
            for line in in_f:
                tmp = line.strip().split(" ", 1)
                uid = tmp[0]
                events = tmp[1]
                user_events[uid] = events

    return user_events


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

                
# Train fastText models
def train(vimp, cv, user_events, emb_v1, emb_v2):
    models = {}
    for cid in cv:
        # Construct training data
        pos = list(set(cv[cid]))
        neg = list(set(vimp[cid]))
        # Prepare training file
        tr_data = []
        for i, a in enumerate(pos):
            tr_data.append('label_1' + '\t' + user_events[a])
        for i, a in enumerate(neg):
            tr_data.append('label_0' + '\t' + user_events[a])

        random.shuffle(tr_data)
        with open('tr.csv', 'w') as out_f:
            for line in tr_data:
                out_f.write(line + '\n')

        # Train model
        model1 = ft.train_supervised(
            input='tr.csv', 
            epoch=25, 
            lr=1.0, 
            wordNgrams=2, 
            verbose=2, 
            minCount=1,

        )

        model2 = ft.train_supervised(
            input='tr.csv', 
            epoch=25, 
            lr=1.0, 
            wordNgrams=2, 
            verbose=2, 
            minCount=1,

        )

        # Store them
        models[cid] = [model1, model2]

    return models


# Apply fitted models to test data
def test(models, vimp, cv, luf, result):
    with open(os.path.join('result', result), "w") as output_file:
        for cid in cv:
            # Construct test data
            pos = list(set(cv[cid]))
            neg = list(set(vimp[cid]))
            # Prepare test file
            X, y_true = [], []
            for i, a in enumerate(pos):
                X.append(user_events[a])
                y_true.append('label_1')
            for i, a in enumerate(neg):
                X.append(user_events[a])
                y_true.append('label_0')
            # Load trained models
            pipe_f, pipe_d = models[cid]
            z_f = pipe_f.predict(X)
            z_d = pipe_d.predict(X)
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
    parser = argparse.ArgumentParser("python3 train_ft.py")
    parser.add_argument("input_data", type=str, help="Input data")
    parser.add_argument("user_events_file", type=str, help="User history events file prefix")
    parser.add_argument("result", type=str, help="Result")
    parser.add_argument("emb_v1", type=str, help="User emb_v1")
    parser.add_argument("emb_v2", type=str, help="User emb_v2")
    args = parser.parse_args()
    print(f"[{get_t()}] reading user events")
    user_events = get_user_events(args.user_events_file)
    print(f"[{get_t()}] reading data")
    vimp1, vimp2, cv1, cv2 = read_data(args.input_data, set(user_events.keys()))
    print(f"[{get_t()}] training")
    models = train(vimp1, cv1, user_events, args.emb_v1, args.emb_v2)
    print(f"[{get_t()}] test")
    # test(models, vimp2, cv2, luf, args.result)
    print(f"[{get_t()}] done")

