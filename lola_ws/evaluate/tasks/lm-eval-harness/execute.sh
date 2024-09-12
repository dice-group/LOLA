#!/bin/bash
. task.config

# export CUDA_VISIBLE_DEVICES=1
slurm=false

# # Parse the commandline args into models, sub tasks and languages
# Sample usage: bash execute.sh -m model_id -s subtask -l language -r absolute_path_result_dir -c
# Example: bash execute.sh -m dice-research/lola_v1 -s xnli -l de -r /data/kshitij/LOLA-Megatron-DeepSpeed/lola_ws/evaluate/tasks/lm-eval-harness/Results -c
# Not using the flag will give an error if model_id, subtask, language and result_path are not specified
# The 'c' is to set the slurm flag 

while getopts ":m:f:n:r:c" opt; do
  case $opt in
    m) model="$OPTARG"
    ;;
    f) final_task_id="$OPTARG"
    ;;
    n) num_few_shot="$OPTARG"
    ;;
    r) result_path="$OPTARG"
    ;;
    c) slurm=true
    ;;
    \?) echo "Invalid option -$OPTARG" >&3
    exit 1
    ;;
  esac

  if [[ $opt != c && -z $OPTARG ]]; then
    echo "Option -$opt requires an argument" >&2
    exit 1
  fi
done


# Activate the virtual environment
CONDA_VENV_DIR=$(realpath ./$TASK_NAME-eval)
source activate ./$TASK_NAME-eval

# the statement below is required on slurm
if [[ "$slurm" == true ]]; then
  echo "Setting LD_LIBRARY_PATH for noctua2."
  export LD_LIBRARY_PATH=$CONDA_VENV_DIR/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
fi


# export this variable to your environment before running this script: export HF_LOLA_EVAL_AT=<your-access-token-here>
huggingface-cli login --token $HF_LOLA_EVAL_AT

# change batch_size to auto:4 to optimize GPU usage. Leave it at 1 for safe/debug mode.
lm_eval --model hf \
    --model_args pretrained="${model}" \
    --tasks $final_task_id \
    --num_fewshot $num_few_shot \
    --device cuda:0 \
    --batch_size 1 \
    --log_samples \
    --output_path "${result_path}" \
    --trust_remote_code