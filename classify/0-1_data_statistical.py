#!/usr/bin/env python3
import os
import glob
import math
import argparse
from tqdm import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

def statistic(data_file, out_folder, allow_campaign_type):
    vimp1, vimp2 = defaultdict(set), defaultdict(set)
    cv1, cv2 = defaultdict(set), defaultdict(set)

    with open(data_file, 'r') as in_f, \
            open(os.path.join(out_folder, 'statistic_data.csv'), 'w') as out_f, \
            open(os.path.join(out_folder, 'data.csv'), 'w') as out_f1:
        for line in in_f:
            tokens = line.strip().split("\t")
            type_id = tokens[0].strip()

            if len(tokens) < 7 or type_id not in allow_campaign_type:
                continue
            out_f1.write('\t'.join(tokens[1:]) + '\n')

            cid, ad_id = tokens[1], tokens[2]
            vimp1_flag, vimp2_flag = int(tokens[3]), int(tokens[4])
            cv1_flag, cv2_flag = int(tokens[5]), int(tokens[6])

            if vimp1_flag > 0:
                vimp1[cid].add(ad_id)
            elif vimp2_flag > 0:
                vimp2[cid].add(ad_id)
            elif cv1_flag > 0:
                cv1[cid].add(ad_id)
            elif cv2_flag > 0:
                cv2[cid].add(ad_id)
            else:
                raise Exception("data error")

        # Just in case
        cids = set(vimp1) & set(vimp2) & set(cv1) & set(cv2)
        out_f.write("Total valid campaign #IDs : {} \n".format(len(cids)))
        out_f.write("\n")
        out_f.write("CampaignID\t#Pos(tr)\t#Neg(tr)\t#Pos(te)\t#Neg(te)\t#Interact_User\n")
        user_ids = set()
        for cid in cids:
            tr_pos = len(cv1[cid])
            tr_neg = len(vimp1[cid])
            te_pos = len(cv2[cid])
            te_neg = len(vimp2[cid])
            user_ids |= (cv1[cid] | vimp1[cid] | cv2[cid] | vimp2[cid])
            user_interact = len((cv1[cid] | vimp1[cid]) & (cv2[cid] | vimp2[cid]))
            out_f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(cid, tr_pos, tr_neg, te_pos, te_neg, user_interact))
        return user_ids


def prepare_user_active(user_ids, out_folder, user_active_f):
    user_active_weight = {}

    with open(user_active_f, 'r') as in_f:
        for line in in_f:
            tmp = line.strip().split("\t")
            uid = tmp[0]
            count = int(tmp[1])
            user_active_weight[uid] = 1 + math.log(count)

    max_value = max(list(user_active_weight.values()))

    with open(os.path.join(out_folder, 'user_active_weight.csv'), 'w') as out_f:
        for uid in user_active_weight:
            if uid in user_ids:
                print(f"{uid}\t{user_active_weight[uid]/max_value:.6f}", file=out_f)


def prepare_shrink_user_embedding(user_ids, out_folder, user_embdding_list):
    for emb_file in user_embdding_list:
        name = os.path.basename(emb_file)
        print('---- processing [{}] file'.format(name))
        with open(emb_file, 'r') as in_f, \
            open(os.path.join(out_folder, name), 'w') as out_f:
            for line in tqdm(in_f):
                user_id = line.split('\t')[0]
                if user_id in user_ids:
                    out_f.write(line)


def get_allow_type(campaign_type):
     # Prepare statistic file
    if campaign_type == 'app':
        allow_campaign_type = ['appstore']
    elif campaign_type == 'web':
        allow_campaign_type = ['webview']
    else:
        allow_campaign_type = ['webview', 'appstore']
    return allow_campaign_type



if __name__ == '__main__':
    parser = argparse.ArgumentParser("python3 0_data_statistical.py")
    parser.add_argument("input_data", type=str, help="Input data")
    parser.add_argument("data_name", type=str, help="Data name (output folder name)")
    parser.add_argument("--user_active_files", type=str, help="user active file list")
    parser.add_argument("--user_embdding_files", type=str, nargs='+', help="user embedding file list", default=[])
    # parser.add_argument("--user_emb", type=str, default=None)
    args = parser.parse_args()

    # Create output folder
    out_folder = os.path.join(CURRENT_PATH, 'train_data', args.data_name)
    os.makedirs(out_folder, exist_ok=True)

    print('1. statistic ...')
    user_ids = set()
    for _type in ['all', 'web', 'app']:
        type_folder = os.path.join(out_folder, _type)
        os.makedirs(type_folder, exist_ok=True)
        user_ids |= statistic(args.input_data, type_folder, get_allow_type(_type))
    with open(os.path.join(out_folder, 'user.list'), 'w') as out_f:
        for uid in user_ids:
            out_f.write(uid + '\n')

    # Prepare user_active file
    print('2. Process user active file : ', args.user_active_files)
    prepare_user_active(user_ids, out_folder, args.user_active_files)

    # Prepare user_events / user_emb file
    print('3. Process embeddings file : ', args.user_embdding_files)
    prepare_shrink_user_embedding(user_ids, out_folder, args.user_embdding_files)



