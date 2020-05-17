#!/bin/bash
dt="2020-04-30"
# For VAE
./data/s3_scripts/fetch_table_data.sh ./data/${dt}/vae_user_features \
			smartad-dmp/warehouse/ml/tmp_fasttext_training_set/dt=${dt}/
./data/s3_scripts/fetch_table_data.sh ./data/${dt}/vae_features_freq \
			smartad-dmp/warehouse/ml/tmp_fasttext_tag_frequency/dt=${dt}/
bulk_data="./data/${dt}/vae_user_features/merged.data"
data_prefix="./data/${dt}/vae_user_features/tags_"
vocab="./data/${dt}/vae_features_freq/merged.data"
split -l 10000 ${bulk_data} ${data_prefix}
rm -f ${bulk_data}
# For AE
./data/s3_scripts/fetch_table_data.sh ./data/${dt}/ae_user_features \
			smartad-dmp/warehouse/ml/exp_libsvm/type=deep_lookalike/dt=${dt}/

outdir="./data/model/"
mkdir -p ${outdir}
# Train VAE
./unsupervised_embedding/train_VAE.py ${data_prefix} ${vocab} ${outdir}/${dt}.luf_vae.vec

# Train AE
data="./data/${dt}/ae_user_features/merged.data"
./unsupervised_embedding/train_AE.py ${data} ${outdir} 18 1000
mv ${outdir}luf.vec ${outdir}/${dt}.luf_ae.vec