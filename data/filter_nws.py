#!/usr/bin/env python3
import os, sys
import errno
from tqdm import tqdm

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def handler(in_dir, out_dir):
    file_list = os.listdir(in_dir)
    for f_name in tqdm(file_list):
        in_file = os.path.join(in_dir, f_name)
        out_file = os.path.join(out_dir, f_name)

        with open(in_file, 'r') as in_f, open(out_file, 'w') as out_f:
            for line in in_f:
                tmp = line.strip().split(' ')
                user_id = tmp[0]
                records = []
                for r in tmp[1:]:
                    if not r.startswith('nws:'):
                        records.append(r)
                if records:
                    out_f.write(user_id + ' ' + ' '.join(records) + "\n")



if __name__ == '__main__':
    print("parameters : ", sys.argv)
    if len(sys.argv) != 3:
        print('please input in_folder & out_folder')
        sys.exit(1)
    in_folder = sys.argv[1]
    out_folder = sys.argv[2]
    mkdir_p(out_folder)
    handler(in_folder, out_folder)
