#!/usr/bin/env python3
import os
import sys
from tqdm import tqdm


count_list = []
def handler(in_dir):
    file_list = os.listdir(in_dir)
    out_put_dir = os.path.join(os.path.dirname(in_dir), '../')
    print("Output Dir : ", out_put_dir)
    out_file = os.path.join(out_put_dir, 's2v_tr.txt')
    with open(out_file, 'w') as out_f:
        for f_name in tqdm(file_list):
            in_file = os.path.join(in_dir, f_name)

            with open(in_file, 'r') as in_f:
                for line in in_f:
                    tmp = line.strip().split(' ')
                    user_id = tmp[0]
                    records = []
                    for r in tmp[1:]:
                        #if not r.startswith('nws:'):
                        records.append(r)
                    if records:
                        out_f.write(' '.join(records) + "\n")
                        count_list.append(len(records))
    print("max_len : {}, min_len : {}".format(max(count_list), min(count_list)))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[Usage] ./get_s2v_data.py [fearures_dir]")
        sys.exit(1)
    in_folder = sys.argv[1]
    handler(in_folder)
