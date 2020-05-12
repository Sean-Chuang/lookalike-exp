#!/usr/bin/env python3
import sys
import numpy as np


n1, n2 = 0, 0
x1, x2 = [], []
y1, y2 = [], []
cnt1 = np.zeros(4, dtype=np.int64)
cnt2 = np.zeros(4, dtype=np.int64)
for line in open(sys.argv[1]):
    tokens = line.strip().split("\t")
    i1 = float(tokens[9])
    i2 = float(tokens[10])
    j1 = float(tokens[11])
    j2 = float(tokens[12])
    h = [int(k) for k in tokens[1:9]]
    cnt1 += h[0:4]
    cnt2 += h[4:9]
    x1.append(i1)
    x2.append(i2)
    y1.append(j1)
    y2.append(j2)
    if j1 > j2:
        n1 += 1
    else:
        n2 += 1

z1, z2 = np.array(x1), np.array(x2)
w1, w2 = np.array(y1), np.array(y2)

print(f"emb_v1 wins = {n1}, emb_v2 wins = {n2}")
print(f"emb_v1 F1: mean = {np.mean(z1):.3f}, sd = {np.std(z1):.3f}")
print(f"emb_v2 F1: mean = {np.mean(z2):.3f}, sd = {np.std(z2):.3f}")
print(f"emb_v1 AP: mean = {np.mean(w1):.3f}, sd = {np.std(w1):.3f}")
print(f"emb_v2 AP: mean = {np.mean(w2):.3f}, sd = {np.std(w2):.3f}")
print(f"emb_v1 [TP TN FP FN]: {cnt1}")
print(f"emb_v2 [TP TN FP FN]: {cnt2}")

