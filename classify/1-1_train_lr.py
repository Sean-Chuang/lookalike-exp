#!/usr/bin/env python3
import os, sys
from tqdm import tqdm
import argparse
import numpy as np

import pickle
from datetime import datetime

from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
from sklearn.metrics import classification_report

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# Read user events
def get_user_events(user_events_file):
    user_events = {}

    with open(user_events_file, 'r') as in_f:
        for line in in_f:
            tmp = line.strip().split('\t')
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

                
# Train LR models
def train(vimp, cv, user_features):
    dim = next(iter(user_features.values())).shape[1]
    models = {}
    for cid in tqdm(cv):
        # Construct training data
        pos = list(set(cv[cid]))
        neg = list(set(vimp[cid]))
        
        # Prepare train data
        train_X = lil_matrix((len(pos) + len(neg), dim))
        train_y = np.zeros(len(pos) + len(neg))

        train_y[:len(pos)] = 1.0
        for i, uid in enumerate(pos):
            train_X[i, :] = user_features[uid]
        for i, uid in enumerate(neg):
            train_X[len(pos)+i, :] = user_features[uid]

        # Set up parameters
        # To follow the convention in spark.ml, C = 1 / (n * lambda)
        lr_param = {
            "random_state": 0,
            "C": 1.0 / (0.3 * len(train_y)), 
            "class_weight": "balanced"
        }
        # Construct pipelines
        lr_model = LogisticRegression(**lr_param)
        # Train them
        lr_model.fit(train_X, train_y)
        # Store them
        models[cid] = lr_model
    return models


# Apply fitted models to test data
def test(models, vimp, cv, user_features, result):
    dim = next(iter(user_features.values())).shape[1]
    with open(os.path.join('result', result), "w") as output_file:
        for cid in tqdm(cv):
            # Construct training data
            pos = list(set(cv[cid]))
            neg = list(set(vimp[cid]))
            # Prepare test data
            test_X = lil_matrix((len(pos) + len(neg), dim))
            test_y = np.zeros(len(pos) + len(neg))

            test_y[:len(pos)] = 1.0
            for i, uid in enumerate(pos):
                test_X[i, :] = user_features[uid]
            for i, uid in enumerate(neg):
                test_X[len(pos)+i, :] = user_features[uid]

            # Load trained models
            lr_model = models[cid]
            predict_y = lr_model.predict(test_X)
            # Count results
            confusion = confusion_matrix(test_y, predict_y).flatten().tolist()
            # result = [np.count_nonzero(test_y * z),
            #             np.count_nonzero((1.0 - test_y) * (1.0 - z)),
            #             np.count_nonzero((1.0 - test_y) * z),
            #             np.count_nonzero(test_y * (1.0 - z))]

            # Calculate F1
            f1 = f1_score(test_y, predict_y)
            # Calculate AP
            sc = lr_model.decision_function(test_X)
            ap = average_precision_score(test_y, sc)
            # Print result
            s_result = "\t".join(map(str, [confusion[0], confusion[3], confusion[1], confusion[2]]))
            print(f"{cid}\t{s_result}\t{f1:.6f}\t{ap:.6f}", file=output_file)


# Time function
def get_t():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("python3 1-1_train_lr.py")
    parser.add_argument("input_data", type=str, help="Input data")
    parser.add_argument("result", type=str, help="Result")
    parser.add_argument("user_events", type=str, help="User history events")
    parser.add_argument("user_features_pickle", type=str, help="User features pickle file")
    args = parser.parse_args()
    print(f"[{get_t()}] reading user events")
    user_events = get_user_events(args.user_events)
    print(f"[{get_t()}] extract user features")
    with open(args.user_features_pickle, 'rb') as handle:
        user_features = pickle.load(handle)
    print(f"[{get_t()}] reading data")
    vimp1, vimp2, cv1, cv2 = read_data(args.input_data, set(user_features.keys()))
    print(f"[{get_t()}] training")
    models = train(vimp1, cv1, user_features)
    print(f"[{get_t()}] test")
    test(models, vimp2, cv2, user_features, args.result)
    print(f"[{get_t()}] done")

