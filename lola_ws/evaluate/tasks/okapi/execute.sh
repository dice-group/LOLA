#!/bin/bash

set -eu

. task.config

# export CUDA_VISIBLE_DEVICES=1
slurm=false

# # Parse the commandline args into models, sub tasks and languages
# Sample usage: bash execute.sh -m model_id -s subtask -l language -r absolute_path_result_dir -c
# Example: bash execute.sh -m dice-research/lola_v1 -l de -r /data/kshitij/LOLA-Megatron-DeepSpeed/lola_ws/evaluate/tasks/okapi/Results -c
# Not using the flag will give an error if model_id, language and result_path are not specified
# The 'c' is to set the slurm flag 
# subtask flag is not necessary and will be ignored even if its provided
while getopts ":m:s:l:r:c:" opt; do
  case $opt in
    m) model="$OPTARG"
    ;;
    s) sub_task="$OPTARG"
    ;;
    l) lang="$OPTARG"
    ;;
    r) result_path="$OPTARG"
    ;;
    c) slurm=true
    ;;
    \?) echo "Invalid option -$OPTARG" >&3
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

# the statement below is required on slurm
if [[ "$slurm" == true ]]; then
  echo "It worked"
  export LD_LIBRARY_PATH=$CONDA_VENV_DIR/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
fi

# Activate the virtual environment
CONDA_VENV_DIR=$(realpath ./$TASK_NAME-eval)
source activate $CONDA_VENV_DIR


make_dir() {
	delimiter="/"
	gen_path="/"
	declare -a path_array=($(echo $1 | tr "$delimiter" " "))
	for dir in "${path_array[@]}"
	do
		gen_path="$gen_path/$dir"
		if [[ ! -d "$gen_path" ]]; then
			mkdir -p $gen_path
		fi
	done
	
}

result_path="$result_path/okapi/$(cut -d'/' -f2 <<<$model)/$lang"
make_dir $result_path
json_path="$result_path/output_$(date +%s).json"

# python3 edits.py '2' $REPO_DIR $result_path
cd $REPO_DIR
# "/data/kshitij/LOLA-Megatron-DeepSpeed/lola_ws/evaluate/tasks/okapi/Results"
bash scripts/run.sh $lang $model $json_path > "${result_path}/output_$(date +%s).txt"

# -u -m debugpy --wait-for-client --listen 0.0.0.0:12121


