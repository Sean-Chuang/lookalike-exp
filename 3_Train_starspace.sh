#!/bin/bash

# Step 1: prepare data


# Step 2: train
./train.sh

# Step 3: fetch user embedding (StarSpace/python/)
./get_ss_emb.py ../sp_2020-04-30 ../../lookalike-exp/data/2020-04-30/vae_user_features/tags_ ../../lookalike-exp/classify/train_data/t_0501/user.list ../../lookalike-exp/classify/train_data/t_0501/ss.emb.vec





./s2v_fetch_vector.py ../data/model/2020-05-06_s2v_model.vec ../data/2020-05-06/vae_user_features/tags_  ../data/2020-05-06/vae_features_freq/merged.data ../classify/train_data/t_0507/user.list ../data/model/2020-05-06_s2v.idf.vec




# Train FT
cd ~
bash smart-ad-dmp/modules/box_v1/ec2/conda_setup.sh
export PATH="$HOME/miniconda/bin:$PATH"

ft_ver="0.9.1"
wget https://github.com/facebookresearch/fastText/archive/v${ft_ver}.zip
unzip v${ft_ver}.zip
cd fastText-${ft_ver}
make

dt="2020-05-06"
./data/s3_scripts/fetch_table_data.sh ../fastText-0.9.1/data/${dt}/vae_user_features \
			smartad-dmp/warehouse/ml/tmp_fasttext_training_set/dt=${dt}/
./data/s3_scripts/fetch_table_data.sh ../fastText-0.9.1/data/${dt}/vae_features_freq \
			smartad-dmp/warehouse/ml/tmp_fasttext_tag_frequency/dt=${dt}/

# sort out tokens in data
data="/mnt1/train/fastText-0.9.1/data/${dt}/vae_user_features/merged.data"
ids="/mnt1/train/fastText-0.9.1/data/${dt}/ids.data"
sentences="/mnt1/train/fastText-0.9.1/data/${dt}/sentences.data"

cd ~/smart-ad-dmp/azkaban-flow/audience
python -u scripts/fasttext_lookalike_tokenize.py ${data} ${ids} ${sentences}

# 進入 ft folder
cd /mnt1/train/
mkdir -p "/mnt1/train/fastText-0.9.1/model/"
model="/mnt1/train/fastText-0.9.1/model/fasttext_lookalike.${dt}.model"
./fastText-0.9.1/fasttext skipgram -thread 32 -ws 5 -minn 0 -maxn 0 -dim 128 -minCount 1 -wordNgrams 1 -epoch 25 -input ${sentences} -output ${model}
du -khs ${model}.vec

# calcualte user embedding by AVG
tag="t_0507"
cd /mnt1/train/lookalike-exp/unsupervised_embedding/
./s2v_fetch_vector.py \
	/mnt1/train/fastText-0.9.1/model/fasttext_lookalike.${dt}.model.vec \
	../../fastText-0.9.1/data/${dt}/vae_user_features/merged.data \
	../../fastText-0.9.1/data/${dt}/vae_features_freq/merged.data  \
	../classify/train_data/${tag}/user.list \
	../classify/train_data/${tag}/fasttext_lookalike.noidf.vec

 ./classify/0-1_data_statistical.py data/data/${tag}/merged.data ${tag}
 ./classify/1-2_train_lr.py \
		./classify/train_data/${tag}/app/data.csv \
		luf.org.${tag}.result \
		./classify/train_data/${tag}/fasttext_lookalike.noidf.vec
./classify/2_count_res.py result/luf.org.${tag}.result


create external table if not exists z_chuang.ad_deeplal_features (
ad_id_plus string,
luf string
)
partitioned by (type string, dt string)
row format delimited fields terminated by '\t'
lines terminated by '\n'
stored as textfile
location 's3://smartad-dmp/warehouse/ml/ad_deeplal_features'
;
