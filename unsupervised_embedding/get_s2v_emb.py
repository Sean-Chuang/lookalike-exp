#!/usr/bin/env python3
import os
import glob
from tqdm import tqdm
import sent2vec
import argparse

model = sent2vec.Sent2vecModel()

def fetch_embedding(model_file, data_prefix, output_file):
    model.load_model(model_file)

    file_list = glob.glob(data_prefix + '*')
    with open(output_file, 'w') as out_f:
        for file_name in tqdm(file_list):
            with open(file_name, 'r') as in_f:
                for line in in_f:
                    tmp = line.strip().split(' ', 1)
                    user_id = tmp[0]
                    vector = model.embed_sentence(tmp[1])[0]
                    vector_str = [str(i) + ':' + f'{v:.3f}' for i,v in enumerate(vector)]
                    out_f.write(user_id + '\t' + ' '.join(vector_str) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("python3 s2v_fetch_vector.py")
    parser.add_argument("model", type=str)
    parser.add_argument("data_prefix", type=str, help="User histroy events file prefix")
    parser.add_argument("embed_path", type=str, help="Output embedding path")
    args = parser.parse_args()
    fetch_embedding(args.model, args.data_prefix, args.embed_path)
