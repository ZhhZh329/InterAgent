export HF_HOME=/Users/zhzhou/Desktop/InterAgent/.cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export XDG_CACHE_HOME="/Users/zhzhou/Desktop/InterAgent/.cache"
export KAGGLE_CONFIG_DIR="/Users/zhzhou/Desktop/InterAgent/.cache/kaggle"

mlebench prepare -c spooky-author-identification
mlebench prepare -c random-acts-of-pizza
mlebench prepare -c nomad2018-predict-transparent-conductors
mlebench prepare -c text-normalization-challenge-english-language
mlebench prepare -c text-normalization-challenge-russian-language

mlebench prepare -c aerial-cactus-identification
mlebench prepare -c leaf-classification
mlebench prepare -c denoising-dirty-documents
mlebench prepare -c jigsaw-toxic-comment-classification-challenge
