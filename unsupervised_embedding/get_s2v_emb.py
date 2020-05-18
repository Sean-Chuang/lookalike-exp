#!/usr/bin/env python3
import os
import glob
from tqdm import tqdm
import sent2vec
import argparse

model = sent2vec.Sent2vecModel()

def get_user_list(user_list_file):
    user_set = set()
    with open(user_list_file, 'r') as in_f:
        for line in in_f:
            user_set.add(line.strip())
    return user_set

def fetch_embedding(model_file, data_prefix, user_list, output_file):
    model.load_model(model_file)

    file_list = glob.glob(data_prefix + '*')
    with open(output_file, 'w') as out_f:
        for file_name in tqdm(file_list):
            with open(file_name, 'r') as in_f:
                for line in in_f:
                    tmp = line.strip().split(' ', 1)
                    user_id = tmp[0]
                    if user_id in user_list:
                        vector = model.embed_sentence(tmp[1])[0]
                        vector_str = [str(i) + ':' + f'{v:.4f}' for i,v in enumerate(vector)]
                        out_f.write(user_id + '\t' + ' '.join(vector_str) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("python3 s2v_fetch_vector.py")
    parser.add_argument("model", type=str)
    parser.add_argument("data_prefix", type=str, help="User histroy events file prefix")
    parser.add_argument("user_list", type=str, help="Avaliable user in test data")
    parser.add_argument("embed_path", type=str, help="Output embedding path")
    args = parser.parse_args()
    print('0. Get user list')
    user_list = get_user_list(args.user_list)
    print('1. Get user embedding')
    fetch_embedding(args.model, args.data_prefix, user_list, args.embed_path)
