#!/bin/bash
. task.config


# # Parse the commandline args into models, sub tasks and languages
# Sample usage: bash execute.sh -m model_id -s subtask -l language -r result_directory
# Example: bash execute.sh -m dice-research/lola_v1 -l de -r Results
# Not using the flag will set default value in case of result directory, 
# will give an error if model_id and language are not specified
# subtask flag is not necessary and will be ignored even if its provided
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


if [[ ! $result_dir ]]; then
    result_dir="Experiment_results"
fi

if [ -d "$result_dir" ]; then
  result_dir="${result_dir}_$(date +%s)"
fi


mkdir "$result_dir"
model_dir="$(cut -d'/' -f2 <<<$model)"
mkdir "${result_dir}/$model_dir"
mkdir "${result_dir}/${model_dir}/$lang"
cd $REPO_DIR


bash scripts/run.sh $lang $model > "../${result_dir}/${model_dir}/${lang}/output.txt"


