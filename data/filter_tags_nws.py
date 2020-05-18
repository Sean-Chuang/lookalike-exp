#!/usr/bin/env python3
import sys
import os
base_dir = sys.argv[1]
in_file = os.path.join(base_dir, 'merged.data')
out_file = os.path.join(base_dir, 'merged.data.no.nws')

with open(in_file, 'r') as in_f, open(out_file, 'w') as out_f:
    for line in in_f:
        tokens = line.strip().split('\t')
        if len(tokens) < 2:
            continue 
        if not tokens[0].startswith('nws:'):
            out_f.write(line)
