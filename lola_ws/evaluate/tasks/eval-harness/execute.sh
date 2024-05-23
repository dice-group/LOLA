#!/bin/bash
. task.config

# Parse the commandline args into models, sub tasks and languages
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
# source eleuther_env/bin/activate
conda init
conda activate $TASK_NAME-eval


# Multi model/subtasks/language support
# Separate strings into arrays
# delimiter=", "
# declare -a models_array=($(echo $models | tr "$delimiter" " "))

# declare -a sub_tasks_array=($(echo $sub_tasks | tr "$delimiter" " "))

# declare -a langs_array=($(echo $langs | tr "$delimiter" " "))


mkdir "$result_dir"
mkdir "${result_dir}/$model"
sub="${sub_task}_$lang"
mkdir "${result_dir}/${model}/$sub"
cd $REPO_DIR

lm_eval --model hf \
    --model_args "${model}" \
    --tasks "${sub}" \
    --device cuda:0 \
    --batch_size 8
    --trust_remote_code > "../${result_dir}/${model}/${sub}/output.txt"



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


