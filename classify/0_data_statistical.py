#!/usr/bin/env python3
import os
import glob
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
        out_f.write("CampaignID\t#Positive\t#Negitive\t#Interact_User\n")
        user_ids = set()
        for cid in cids:
            pos = len(cv1[cid])
            neg = len(vimp1[cid])
            user_ids |= (cv1[cid] | vimp1[cid] | cv2[cid] | vimp2[cid])
            user_interact = len((cv1[cid] | vimp1[cid]) & (cv2[cid] | vimp2[cid]))
            out_f.write('{}\t{}\t{}\t{}\n'.format(cid, pos, neg, user_interact))
        return user_ids


def prepare_shrink_data(user_ids, user_events_prefix, out_folder, out_file_prefix):
    events_file = os.path.join(out_folder, out_file_prefix + 'user_events.csv')
    user_events = {}
    file_list = glob.glob(user_events_prefix + '*')
    for file_name in tqdm(file_list):
        with open(file_name, 'r') as in_f:
            for line in in_f:
                tmp = line.strip().split(" ", 1)
                uid = tmp[0]
                events = tmp[1]
                if uid in user_ids:
                    user_events[uid] = events

    with open(events_file, 'w') as out_f:
        for uid in user_events:
            out_f.write(uid + '\t' + user_events[uid] + '\n')

    return user_events


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



def get_events_features(user_events, out_folder, out_file_prefix):
    vectorizer = TfidfVectorizer(
                    max_features=20000,
                    analyzer='word',
                    sublinear_tf=True,
                    dtype=np.float32
                )
    keys = list(user_events.keys())
    values = list(user_events.values())
    features = vectorizer.fit_transform(values)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = features[idx]
    with open(os.path.join(out_folder, out_file_prefix + 'features.pickle'), 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(out_folder, out_file_prefix + 'vectorizer.pickle'), 'wb') as handle:
        pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("python3 0_data_statistical.py")
    parser.add_argument("input_data", type=str, help="Input data")
    parser.add_argument("data_name", type=str, help="Data name (output folder name)")
    parser.add_argument("user_events_prefix", type=str, help="User events file prefix")
    parser.add_argument("--file_prefix", type=str, help="output file prefix", default=None)
    parser.add_argument("--campaign_type", type=str, help="campaign type (app/web/all)", default='all')
    parser.add_argument("--user_embdding_files", type=str, nargs='+', help="user embedding file list", default=[])
    # parser.add_argument("--user_emb", type=str, default=None)
    args = parser.parse_args()

    # Create output folder
    out_folder = os.path.join(CURRENT_PATH, 'train_data', args.data_name, args.campaign_type)
    os.makedirs(out_folder, exist_ok=True)
    file_prefix = args.file_prefix + '_' if args.file_prefix else ''
    # Prepare statistic file
    if args.campaign_type == 'app':
        allow_campaign_type = ['appstore']
    elif args.campaign_type == 'web':
        allow_campaign_type = ['webview']
    else:
        allow_campaign_type = ['webview', 'appstore']
    print('Allow campagin type', allow_campaign_type)
    print('1. statistic ...')
    user_ids = statistic(args.input_data, out_folder, allow_campaign_type)
    # Prepare user_events / user_emb file
    print('2. Process shrink data ...')
    user_events = prepare_shrink_data(user_ids, args.user_events_prefix, out_folder, file_prefix)
    print('3. Process embeddings file : ', args.user_embdding_files)
    prepare_shrink_user_embedding(user_ids, out_folder, args.user_embdding_files)
    # Prepare tfidf features
    print('4. Get event tfidf features')
    get_events_features(user_events, out_folder, file_prefix)


