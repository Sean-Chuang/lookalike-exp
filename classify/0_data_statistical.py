#!/usr/bin/env python3
import os
import glob
import argparse
from tqdm import tqdm
from collections import defaultdict

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

def statistic(data_file, out_folder):
    vimp1, vimp2 = defaultdict(set), defaultdict(set)
    cv1, cv2 = defaultdict(set), defaultdict(set)

    with open(data_file, 'r') as in_f, \
            open(os.path.join(out_folder, 'statistic_data.csv'), 'w') as out_f:
        for line in in_f:
            tokens = line.strip().split("\t")

            if len(tokens) < 6:
                continue
            cid, ad_id = tokens[0], tokens[1]
            vimp1_flag, vimp2_flag = int(tokens[2]), int(tokens[3])
            cv1_flag, cv2_flag = int(tokens[4]), int(tokens[5])

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
            user_ids += (cv1[cid] | vimp1[cid] | cv2[cid] | vimp2[cid])
            user_interact = len((cv1[cid] | vimp1[cid]) & (cv2[cid] | vimp2[cid]))
            out_f.write('{}\t{}\t{}\t{}\n'.format(cid, pos, neg, user_interact))
        return user_ids


def prepare_shrink_data(user_ids, user_events_prefix, out_folder):
    events_file = os.path.join(out_folder, 'user_events.csv')
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser("python3 0_data_statistical.py")
    parser.add_argument("input_data", type=str, help="Input data")
    parser.add_argument("data_name", type=str, help="Data name (output folder name)")
    parser.add_argument("user_events_prefix", type=str, help="User events file prefix")
    # parser.add_argument("--user_emb", type=str, default=None)
    args = parser.parse_args()

    # Create output folder
    out_folder = os.path.join(CURRENT_PATH, 'train_data', args.data_name)
    os.makedirs(out_folder, exist_ok=True)

    # Prepare statistic file
    user_ids = statistic(args.input_data, out_folder)
    # Prepare user_events / user_emb file
    prepare_shrink_data(user_ids, args.user_events_prefix, out_folder)


