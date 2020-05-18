#!/bin/bash
dt=$1
input_file="/mnt1/train/lookalike-exp/data/${dt}/s2v_tr.txt"
echo "intput file ${input_file}"
ft="/mnt1/train/lookalike-exp/unsupervised_embedding/bin/fasttext"
$ft sent2vec -input ${input_file} -output /mnt1/train/lookalike-exp/data/model/${dt}_s2v_model \
	-minCount 3 \
	-dim 128 \
	-epoch 20 \
	-lr 0.15 \
	-wordNgrams 1 \
	-loss ns \
	-neg 20 \
	-thread 20 \
	-t 0.00005 \
	-bucket 2000000 \
	-numCheckPoints 1

