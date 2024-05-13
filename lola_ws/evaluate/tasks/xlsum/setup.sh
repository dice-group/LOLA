#! /bin/bash
# Importing config variables
. task.config


#clone the eval repository
git clone TASK_GIT_REPOSITORY

cd $REP_DIR


#create and activate environment task
conda create --prefix ./$TASK_NAME-eval python=3

conda activate ./$TASK_NAME-eval


#clone the eval repository
git clone $TASK_GIT_REPOSITORY                 

cd $REP_DIR

git checkout $TASK_GIT_HASH

conda create python==3.7.9 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch -p ./env

conda activate ./env # or source activate ./env (for older versions of anaconda)

bash setup.sh


#Extracting the data
wget $DATASET_LINK

tar -xjvf $DATASET_NAME

 python extract_data.py -i XLSum_complete_v2.0/ -o XLSum_input/

echo  'extraction finished'
