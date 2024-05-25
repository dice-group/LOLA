#! /bin/bash
# Importing config variables
. task.config

#SUB_TASK= task_name
#PRETRAINED_MODEL= your_model

#create and activate environment task
yes | conda create --prefix ./$TASK_NAME-eval python=3

source activate ./$TASK_NAME-eval

#clone the eval repository
git clone $TASK_GIT_REPOSITORY $REPO_DIR

cd $REPO_DIR

git checkout $TASK_GIT_HASH

# yes |
pip install -e .

echo  'installation finished'
