#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "[Usage] train_sent2vec.sh [Data_Path] [Out_Model_Name]"
    exit 1
fi

TRAIN_DATA=$1
MODEL_NAME=$2

# Step 1 : Install sent2vec
if [ ! -d "temp/sent2vec" ]; then
    mkdir -p temp
    cd temp
    git clone https://github.com/epfml/sent2vec.git
    cd sent2vec
    make
    pip install .
else
    cd temp/sent2vec
fi

# Step 2 : Training
./fasttext sent2vec -input $TRAIN_DATA -output $MODEL_NAME \
        -minCount 1 \
        -dim 128 \
        -epoch 25 \
        -lr 0.2 \
        -wordNgrams 1 \
        -loss ns \
        -neg 20 \
        -thread 10 \
        -t 0.00005 \
        -bucket 4000000 \
        -numCheckPoints 10


# Step 3 : Get user embedding
# ./get_s2v_emb.py [model] [user_history_data_prefix] [output_vector_file]