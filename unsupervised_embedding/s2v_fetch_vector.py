#!/usr/bin/env python3
import os
import glob
import argparse
from tqdm import tqdm
import numpy as np

def get_user_list(user_list_file):
    user_set = set()
    with open(user_list_file, 'r') as in_f:
        for line in in_f:
            user_set.add(line.strip())
    return user_set
            

def get_user_events(user_events_prefix, user_list):
    user_events = {}
    file_list = glob.glob(user_events_prefix + '*')
    for file_name in tqdm(file_list):
        with open(file_name, 'r') as in_f:
            for line in list(in_f):
                tmp = line.strip().split(" ", 1)
                uid = tmp[0]
                if uid not in user_list:
                    continue
                events = tmp[1]
                user_events[uid] = events
    return user_events


def calculate_idf(vocab_path):
    # read vocabulary
    raw_v = []
    n_total = 0
    stats = {}
    with open(vocab_path, "r") as f:
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

    N = len(raw_v)
    voc = [raw_v[i][0] for i in range(N)]
    frq = [raw_v[i][1] for i in range(N)]
    frq = np.array(frq, dtype=np.float32)
    # find n_max
    n_max = np.max(frq)
    # convert frq to idf
    idf = np.log(1 + n_max / (1 + frq))
    # return vocabulary and stats
    return dict(zip(voc, idf))
    

def read_event_emb(vector_file):
    events_vec = {}
    with open(vector_file, 'r') as in_f:
        for line in tqdm(in_f):
            tmp = line.strip().split(' ')
            e_id = tmp[0]
            vector = np.array(tmp[1:]).astype(np.float)
            events_vec[e_id] = vector / np.linalg.norm(vector)
    return events_vec


def fetch_embedding(idf_map, events_vec, user_events, output_file):
    with open(output_file, 'w') as out_f:
        for user in tqdm(user_events):
            events = user_events[user].split(' ')
            total_w = sum([idf_map[e] for e in events if e in events_vec])
            events = np.array([events_vec[e] * idf_map[e] / total_w for e in events if e in events_vec])
            vector = np.sum(events, 0)
            vector_str = [str(i) + ':' + f'{v:.4f}' for i,v in enumerate(vector)]
            out_f.write(user + '\t' + ' '.join(vector_str) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("python3 s2v_fetch_vector.py")
    parser.add_argument("model_vec", type=str)
    parser.add_argument("data_prefix", type=str, help="User histroy events file prefix")
    parser.add_argument("vocab_freq", type=str, help="User histroy events frequency")
    parser.add_argument("user_list", type=str, help="Avaliable user in test data")
    parser.add_argument("embed_path", type=str, help="Output embedding path")
    args = parser.parse_args()
    print('0. Get user list')
    user_list = get_user_list(args.user_list)
    print('1. Get user events')
    user_events = get_user_events(args.data_prefix, user_list)
    print('2. Get events idf')
    idf_map = calculate_idf(args.vocab_freq)
    print('3. Get events embedding')
    events_vec = read_event_emb(args.model_vec)
    print('4. Get user embedding')
    fetch_embedding(idf_map, events_vec, user_events, args.embed_path)
