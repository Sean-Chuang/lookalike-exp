#!/usr/bin/env python3
import os, sys
from tqdm import tqdm
import argparse
import numpy as np
from datetime import datetime

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
def train(vimp, cv, user_events):
    models = {}
    for cid in tqdm(cv):
        # Construct training data
        pos = list(set(cv[cid]))
        neg = list(set(vimp[cid]))
        
        # Prepare train data
        vectorizer = TfidfVectorizer(
                            max_features=20000,
                            analyzer='word',
                            sublinear_tf=True
                        )

        train_X, train_y = [], []
        for uid in pos:
            train_X.append(user_events[uid])
            train_y.append('pos')
        for uid in neg:
            train_X.append(user_events[uid])
            train_y.append('neg')

        # Set up parameters
        # To follow the convention in spark.ml, C = 1 / (n * lambda)
        lr_param = {"C": 1.0 / (0.3 * len(train_y)), "class_weight": "balanced"}
        # Construct pipelines
        pipe_f = Pipeline([('vect', vectorizer),
                           ('lr', LogisticRegression(**lr_param))])

        # Train them
        pipe_f.fit(train_X, train_y)
        # Store them
        models[cid] = pipe_f
    return models


# Apply fitted models to test data
def test(models, vimp, cv, user_events, result):
    with open(os.path.join('result', result), "w") as output_file:
        for cid in tqdm(cv):
            # Construct training data
            pos = list(set(cv[cid]))
            neg = list(set(vimp[cid]))
            test_X, test_y = [], []
            for uid in pos:
                test_X.append(user_events[uid])
                test_y.append('pos')
            for uid in neg:
                test_X.append(user_events[uid])
                test_y.append('neg')

            # Load trained models
            pipe = models[cid]
            z = pipe.predict(test_X)
            # Count results
            confusion = confusion_matrix(test_y, z).flatten().tolist()
            # result = [np.count_nonzero(test_y * z),
            #             np.count_nonzero((1.0 - test_y) * (1.0 - z)),
            #             np.count_nonzero((1.0 - test_y) * z),
            #             np.count_nonzero(test_y * (1.0 - z))]

            # Calculate F1
            f1 = f1_score(y_true, z_f)
            # Calculate AP
            sc = pipe_f.decision_function(test_X)
            ap = average_precision_score(test_y, sc)
            # Print result
            s_result = "\t".join([str(r) for r in result])
            print(f"{cid}\t{s_result}\t{f1:.6f}\t{ap:.6f}", file=output_file)


# Time function
def get_t():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("python3 1-1_train_lr.py")
    parser.add_argument("input_data", type=str, help="Input data")
    parser.add_argument("result", type=str, help="Result")
    parser.add_argument("user_events", type=str, help="User history events")
    args = parser.parse_args()
    print(f"[{get_t()}] reading user events")
    user_events = get_user_events(args.user_events)
    print(f"[{get_t()}] reading data")
    vimp1, vimp2, cv1, cv2 = read_data(args.input_data, set(user_events.keys()))
    print(f"[{get_t()}] training")
    models = train(vimp1, cv1, user_events)
    print(f"[{get_t()}] test")
    test(models, vimp2, cv2, user_events, args.result)
    print(f"[{get_t()}] done")

