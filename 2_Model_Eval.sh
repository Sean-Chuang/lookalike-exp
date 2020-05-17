#!/bin/bash
dt="2020-05-10"
tag="t_0511"
type="web"

./classify/0-1_data_statistical.py \
				data/data/${tag}/merged.data \
				${tag} --user_embdding_files luf.org.vec luf.new.vec luf.deep.vec
# Test vae
./classify/1-2_train_lr.py \
				./classify/train_data/${tag}/${type}/data.csv \
				luf.org.${tag}.${type}.result \
				./classify/train_data/${tag}/luf.org.vec

# Test vae (no_news)
./classify/1-2_train_lr.py \
				./classify/train_data/${tag}/web/data.csv \
				luf.org.${tag}.web.result \
				./classify/train_data/${tag}/luf.org.vec

# Test ae
./classify/1-2_train_lr.py \
				./classify/train_data/${tag}/app/data.csv \
				luf.org.${tag}.web.result \
				./classify/train_data/${tag}/luf.org.vec