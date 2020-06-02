#!/usr/bin/env python3
import sys
import numpy as np

result_file = sys.argv[1]

f1_score, ap_score, rank_ap_score = [], [], []
cnt = np.zeros(4, dtype=np.int64)
for line in open(result_file):
    tokens = line.strip().split("\t")
    h = [int(k) for k in tokens[1:5]]
    cnt += h[0:4]

    f1_score.append(float(tokens[5]))
    ap_score.append(float(tokens[6]))
    rank_ap_score.append(float(tokens[7]))

f1_score, ap_score, rank_ap_score = np.array(f1_score), np.array(ap_score), np.array(rank_ap_score)

print(f"F1: mean = {np.mean(f1_score):.3f}, sd = {np.std(f1_score):.3f}")
print(f"AP: mean = {np.mean(ap_score):.3f}, sd = {np.std(ap_score):.3f}")
print(f"Rank AP: mean = {np.mean(rank_ap_score):.3f}, sd = {np.std(rank_ap_score):.3f}")
print(f"[TP TN FP FN]: {cnt}")

