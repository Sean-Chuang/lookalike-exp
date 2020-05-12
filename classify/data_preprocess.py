#!/usr/bin/env python3
import os
import argparse
from tqdm import tqdm

"""
Produce 3 kind of data
- user pb data
- user emb_v1
- user emb_v2

This script only keep user_id existed in user pb data.
"""

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(CURRENT_PATH, 'train_data')

def preprocess(input_f, emb_1, emb_2):
    user_ids = set()
    with open(input_f, 'r') as in_f:
        for line in in_f:
            tokens = line.strip().split('\t')
            if len(tokens) < 6:
                continue
            uid = tokens[1]
            user_ids.add(uid)

    keep_uids_v1 = set()
    emb_v1 = dict()
    with open(emb_1, 'r') as in_f:
        for line in tqdm(in_f):
            tmp = line.strip().split("\t")
            uid = tmp[0]
            if uid not in user_ids:
                continue
            keep_uids_v1.add(uid)
            emb_v1[uid] = tmp[1]

    keep_uids_v2 = set()
    emb_v2 = dict()
    with open(emb_2, 'r') as in_f:
        for line in tqdm(in_f):
            tmp = line.strip().split("\t")
            uid = tmp[0]
            if uid not in user_ids:
                continue
            keep_uids_v2.add(uid)
            emb_v2[uid] = tmp[1]

    print("Diff emb_v1 & emb_v2 ", len(keep_uids_v1 - keep_uids_v2), len(keep_uids_v2 - keep_uids_v1))
    interact_user = keep_uids_v1 & keep_uids_v2

    out_file1 = os.path.join(out_dir, 'emb_v1.vec')
    out_file2 = os.path.join(out_dir, 'emb_v2.vec')
    with open(out_file1, 'w') as out_f1, open(out_file2, 'w') as out_f2:
        for uid in tqdm(interact_user):
            out_f1.write(uid + '\t' + emb_v1[uid])
            out_f2.write(uid + '\t' + emb_v2[uid])

    print("Missing user from emb: {}".format(len(user_ids - interact_user)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("python3 data_preprocess.py")
    parser.add_argument("input_data", type=str, help="Input data")
    parser.add_argument("user_emb_v1", type=str, help="User emb_v1")
    parser.add_argument("user_emb_v2", type=str, help="User emb_v2")
    args = parser.parse_args()
    preprocess(args.input_data, args.user_emb_v1, args.user_emb_v2)


