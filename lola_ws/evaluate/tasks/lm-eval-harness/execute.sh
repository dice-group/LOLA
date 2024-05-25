#!/bin/bash
. task.config

# # Parse the commandline args into models, sub tasks and languages
# Sample usage: bash execute.sh -m model_id -s subtask -l language -r result_directory
# Example: bash execute.sh -m dice-research/lola_v1 -s xnli -l de -r Results
# Not using the flag will set default value in case of result directory, 
# will give an error if model_id, subtask and language are not specified

while getopts ":m:s:l:r:" opt; do
  case $opt in
    m) model="$OPTARG"
    ;;
    s) sub_task="$OPTARG"
    ;;
    l) lang="$OPTARG"
    ;;
    r) result_dir="$OPTARG"
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




# Multi model/subtasks/language support
# Separate strings into arrays
# delimiter=", "
# declare -a models_array=($(echo $models | tr "$delimiter" " "))

# declare -a sub_tasks_array=($(echo $sub_tasks | tr "$delimiter" " "))

# declare -a langs_array=($(echo $langs | tr "$delimiter" " "))

if [[ ! $result_dir ]]; then
    result_dir="Experiment_results"
fi

if [[ -d "$result_dir" ]]; then
  result_dir="${result_dir}_$(date +%s)"
fi

mkdir "$result_dir"
model_dir="$(cut -d'/' -f2 <<<$model)"
mkdir "${result_dir}/$model_dir"
sub="${sub_task}_$lang"
mkdir "${result_dir}/${model_dir}/$sub"
cd $REPO_DIR

lm_eval --model hf \
    --model_args pretrained="${model}" \
    --tasks "${sub}" \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code > "../${result_dir}/${model_dir}/${sub}/output.txt"



# Multi model/subtasks/language support
# for model in "${models_array[@]}"
# do
#     mkdir "results/$model"
#     for sub_task in "${sub_tasks_array[@]}"
#     do
#         for lang in "${lang_array[@]}"
#         do 
#             sub="${sub_task}_$lang"
#             mkdir "results/${model}/$sub"
#             chdir "lm-evaluation-harness"
#             lm_eval --model hf \
#             --model_args "${model}" \
#             --tasks "${sub}" \
#             --device cuda:0 \
#             --batch_size 8
#             --trust_remote_code > "../${model}/${sub}/output.txt"
#         done
#     done
# done


