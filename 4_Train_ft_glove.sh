# Prepare evn
cd ~
bash smart-ad-dmp/modules/box_v1/ec2/conda_setup.sh
export PATH="$HOME/miniconda/bin:$PATH"
mkdir /mnt1/train
# - install ft
cd /mnt1/train
wget https://github.com/facebookresearch/fastText/archive/v0.9.1.zip
unzip v0.9.1.zip
cd fastText-0.9.1
make

# - install glove
cd /mnt1/train
git clone http://github.com/stanfordnlp/glove
cd glove
make

# - install lookalike
cd /mnt1/train
git clone https://github.com/Sean-Chuang/lookalike-exp.git

# Download train data and test data
cd /mnt1/train/
dt="2020-05-23"
tag="t_0524_0525"
./lookalike-exp/data/s3_scripts/fetch_table_data.sh tr_data/${dt}/ \
			smartad-dmp/warehouse/ml/tmp_fasttext_training_set/dt=${dt}
./lookalike-exp/data/s3_scripts/fetch_table_data.sh te_data/${tag}/ \
			smartad-dmp/warehouse/user/seanchuang/test_lal_offline_data/dt=${tag}/
./lookalike-exp/data/s3_scripts/fetch_table_data.sh user_active/${dt}/ \
			smartad-dmp/warehouse/user/seanchuang/tmp_user_active_count/dt=${dt}
mkdir "model"

# Process training data
data="/mnt1/train/tr_data/${dt}/merged.data"
ids="/mnt1/train/tr_data/${dt}/ids.data"
sentences="/mnt1/train/tr_data/${dt}/sentences.data"
ft_model="/mnt1/train/model/fasttext_lookalike.${dt}.model"
glove_model="/mnt1/train/model/glove.${dt}.model"

cd ~/smart-ad-dmp/azkaban-flow/audience
python -u scripts/fasttext_lookalike_tokenize.py ${data} ${ids} ${sentences} 


# Train fasttext
cd /mnt1/train/
./fastText-0.9.1/fasttext skipgram -thread 32 -ws 5 -minn 0 -maxn 0 -dim 128 -minCount 1 -wordNgrams 1 -epoch 25 -input ${sentences} -output ${ft_model}
du -khs ${ft_model}.vec

# Train glove
# --- modify the demo parameters
#######################
# CORPUS=${sentences}
# VOCAB_FILE=vocab.txt
# COOCCURRENCE_FILE=cooccurrence.bin
# COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
# BUILDDIR=build
# SAVE_FILE=${glove_model}
# VERBOSE=2
# MEMORY=4.0
# VOCAB_MIN_COUNT=5
# VECTOR_SIZE=128   <----
# MAX_ITER=15
# WINDOW_SIZE=100   <----
# BINARY=2
# NUM_THREADS=30    <----
# X_MAX=10
####################### 
cd /mnt1/train/glove
export sentences glove_model
./demo.sh
du -khs ${glove_model}.txt

# 1. process test data for training LR
cd /mnt1/train
./lookalike-exp/classify/0-1_data_statistical.py te_data/${tag}/merged.data ${tag} \
	--user_active_files=user_active/${dt}/merged.data

# 2. fetch vector (FT)
ft_user_vec="ft.org.te.vec"
./lookalike-exp/unsupervised_embedding/s2v_fetch_vector.py \
	${ft_model}.vec \
	tr_data/${dt}/merged.data \
	None  \
	./lookalike-exp/classify/train_data/${tag}/user.list \
	./lookalike-exp/classify/train_data/${tag}/${ft_user_vec}

#3. train lr
./lookalike-exp/classify/1-2_train_lr.py \
	./lookalike-exp/classify/train_data/${tag}/all/data.csv \
	luf.org.${tag}.result \
	./lookalike-exp/classify/train_data/${tag}/${ft_user_vec}
#4 test lr
./lookalike-exp/classify/2_count_res.py result/luf.org.${tag}.result


# 2. fetch vector (Glove)
glove_user_vec="glove.org.te.vec"
./lookalike-exp/unsupervised_embedding/s2v_fetch_vector.py \
	${glove_model}.txt \
	tr_data/${dt}/merged.data \
	None  \
	./lookalike-exp/classify/train_data/${tag}/user.list \
	./lookalike-exp/classify/train_data/${tag}/${glove_user_vec}

#3. train lr
./lookalike-exp/classify/1-2_train_lr.py \
	./lookalike-exp/classify/train_data/${tag}/all/data.csv \
	luf.org.${tag}.result \
	./lookalike-exp/classify/train_data/${tag}/${glove_user_vec}
#4 test lr
./lookalike-exp/classify/2_count_res.py result/luf.org.${tag}.result
