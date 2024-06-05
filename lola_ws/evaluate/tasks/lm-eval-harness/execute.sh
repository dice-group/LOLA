#!/bin/bash
. task.config

 export CUDA_VISIBLE_DEVICES=1

# # Parse the commandline args into models, sub tasks and languages
# Sample usage: bash execute.sh -m model_id -s subtask -l language -r absolute_path_result_dir
# Example: bash execute.sh -m dice-research/lola_v1 -s xnli -l de -r /data/kshitij/LOLA-Megatron-DeepSpeed/lola_ws/evaluate/tasks/lm-eval-harness/Results
# Not using the flag will give an error if model_id, subtask, language and result_path are not specified

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


# Activate the virtual environment
source activate ./$TASK_NAME-eval

make_dir() {
	delimiter="/"
	gen_path="/"
	declare -a path_array=($(echo $1 | tr "$delimiter" " "))
	for dir in "${path_array[@]}"
	do
		gen_path="$gen_path/$dir"
		if [[ ! -d "$gen_path" ]]; then
			mkdir $gen_path
		fi
	done
	
}


# Ensuring existence/generating results directory in results/task/model/subtask/language support

result_path="$result_path/lm-eval-harness/$(cut -d'/' -f2 <<<$model)/$sub_task/$lang"
make_dir $result_path


lm_eval --model hf \
    --model_args pretrained="${model}" \
    --tasks "${sub_task}_$lang" \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code > "${result_path}/output_$(date +%s).txt"



