#!/bin/bash
. task.config

 export CUDA_VISIBLE_DEVICES=1
slurm=false

# # Parse the commandline args into models, sub tasks and languages
# Sample usage: bash execute.sh -m model_id -s subtask -l language -r absolute_path_result_dir -c
# Example: bash execute.sh -m dice-research/lola_v1 -s xnli -l de -r /data/kshitij/LOLA-Megatron-DeepSpeed/lola_ws/evaluate/tasks/lm-eval-harness/Results -c
# Not using the flag will give an error if model_id, subtask, language and result_path are not specified
# The 'c' is to set the slurm flag 

while getopts ":m:s:l:r:" opt; do
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
source activate ./$TASK_NAME-eval

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


# Huggingface token from Kshitij's account
huggingface_token="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"

# Enter token
expect <<EOF
spawn huggingface-cli login
expect "Enter your token (input will not be visible):"
send "$huggingface_token\r"
expect "Add token as git credential? (Y/n)"
send "yes\r"
expect eof
EOF


# Ensuring existence/generating results directory in results/task/model/subtask/language support

result_path="$result_path/lm-eval-harness/$sub_task/$(cut -d'/' -f2 <<<$model)/$lang"
make_dir $result_path


lm_eval --model hf \
    --model_args pretrained="${model}" \
    --tasks "${sub_task}_$lang" \
    --device cuda:0 \
    --batch_size 8 \
    --log_samples \
    --output_path "${result_path}/results" \
    --trust_remote_code > "${result_path}/output_$(date +%s).txt"

