# Prepare evn
cd ~
bash smart-ad-dmp/modules/box_v1/ec2/conda_setup.sh
export PATH="$HOME/miniconda/bin:$PATH"

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

# Download train data and test data
cd /mnt1/train/
dt="2020-04-30"
./lookalike-exp/data/s3_scripts/fetch_table_data.sh tr_data/${dt}/ \
			smartad-dmp/warehouse/ml/tmp_fasttext_training_set/dt=${dt}/


tag="t_0501_0502"
./lookalike-exp/data/s3_scripts/fetch_table_data.sh te_data/${tag}/ \
			smartad-dmp/warehouse/user/seanchuang/test_lal_offline_data/dt=${tag}/


# Process training data
data="/mnt1/train/tr_data/${dt}/merged.data"
ids="/mnt1/train/tr_data/${dt}/ids.data"
sentences="/mnt1/train/tr_data/${dt}/sentences.data"

cd ~/smart-ad-dmp/azkaban-flow/audience
python -u scripts/fasttext_lookalike_tokenize.py ${data} ${ids} ${sentences} 


# Train fasttext
cd /mnt1/train/
mkdir "model"
model="/mnt1/train/model/fasttext_lookalike.${dt}.model"
./fastText-0.9.1/fasttext skipgram -thread 32 -ws 5 -minn 0 -maxn 0 -dim 128 -minCount 1 -wordNgrams 1 -epoch 25 -input ${sentences} -output ${model}
du -khs ${model}.vec

# Train glove
# --- modify the demo parameters
#######################
# CORPUS=${sentences}
# VOCAB_FILE=vocab.txt
# COOCCURRENCE_FILE=cooccurrence.bin
# COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
# BUILDDIR=build
# SAVE_FILE=glove.${dt}.model
# VERBOSE=2
# MEMORY=4.0
# VOCAB_MIN_COUNT=5
# VECTOR_SIZE=128
# MAX_ITER=15
# WINDOW_SIZE=100
# BINARY=2
# NUM_THREADS=30
# X_MAX=10
####################### 
cd /mnt1/train/
model="/mnt1/train/model/glove.${dt}.model"
./demo.sh
du -khs ${model}.vec

# Test




