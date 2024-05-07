
EVAL_PATH= git clone https://github.com/EleutherAI/lm-evaluation-harness

TASK=EleutherAI
#SUB_TASK= task_name
#PRETRAINED_MODEL= your_model

#create and activate environment task
conda create --prefix ./$TASK-eval python=3

conda activate ./$TASK-eval


#clone the eval repository
git clone $EVAL_PATH

cd lm-evaluation-harness
pip install -e .

echo  'installation finished'
