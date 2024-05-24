#! /bin/bash
# Importing config variables
. task.config

#SUB_TASK= task_name
#PRETRAINED_MODEL= your_model

#create and activate environment task
yes | conda create --prefix ./$TASK_NAME-eval python=3

source ./$TASK_NAME-eval/bin/activate


#clone the eval repository
git clone $TASK_GIT_REPOSITORY $REPO_DIR

cd $REPO_DIR

git checkout $TASK_GIT_HASH

pip install  -e ".[multilingual]"


#download rhe dataset
bash scripts/download.sh

echo  'installation finished'

# edit files to add trust_remote_code
cd ..

python3 edits.py $REPO_DIR

echo  'files adjusted'
