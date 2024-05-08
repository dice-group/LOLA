#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --models=<models> --tasks=<tasks> --languages=<languages> [--results-dir=<dir>]"
    echo "  --models    Comma-separated list of models (e.g., 'model1,model2')"
    echo "  --tasks     Tasks with optional subtasks formatted as 'task1:sub1,sub2;task2'"
    echo "  --languages Comma-separated list of languages (e.g., 'en,es')"
    echo "  --results-dir Optional directory for saving results (default: ./results)"
    echo "Example:"
    echo "  $0 --models=model1,model2 --tasks=\"task1:sub1,sub2;task2\" --languages=en,es --results-dir=./output"
    exit 1
}

# Initialize variables
models=""
tasks=""
languages=""
results_dir="./results"  # Default directory for results

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    usage
fi

for i in "$@"
do
case $i in
    --help)
    usage
    ;;
    --models=*)
    models="${i#*=}"
    shift # past argument=value
    ;;
    --tasks=*)
    tasks="${i#*=}"
    shift # past argument=value
    ;;
    --languages=*)
    languages="${i#*=}"
    shift # past argument=value
    ;;
    --results-dir=*)
    results_dir="${i#*=}"
    shift # past argument=value
    ;;
    *)
    echo "Unknown option: $i"
    usage
    ;;
esac
done

# Validation
if [[ -z "$models" || -z "$tasks" || -z "$languages" ]]; then
    echo "Error: All arguments --models, --tasks, and --languages are required."
    exit 1
fi

# Create the results directory if it does not exist
mkdir -p "$results_dir"

# Parsing tasks to handle subtasks
declare -A taskSubtasksMap
IFS=';' read -ra ADDR <<< "$tasks"
for i in "${ADDR[@]}"; do
    task="${i%%:*}"
    subtasks="${i#*:}"
    if [[ "$subtasks" == "$i" ]]; then
        # No subtasks defined, just the task itself
        subtasks=""
    fi
    taskSubtasksMap["$task"]="$subtasks"
done

# Function to extract languages
get_task_languages() {
    local task_name="$1"
    local subtask_name="$2"
    local json_file="$3"

    # Check if the task contains subtasks
    if jq -e ".tasks[] | select(.name == \"$task_name\") | .subtasks[] | select(.name == \"$subtask_name\")" "$json_file" > /dev/null; then
        # Task and subtask found, extract languages for the subtask
        jq -r ".tasks[] | select(.name == \"$task_name\") | .subtasks[] | select(.name == \"$subtask_name\") | .languages[]" "$json_file"
    elif jq -e ".tasks[] | select(.name == \"$task_name\")" "$json_file" > /dev/null; then
        # Only task found, extract languages directly from the task (if applicable)
        jq -r ".tasks[] | select(.name == \"$task_name\") | .languages[]" "$json_file"
    else
        echo "Task or subtask not found."
        exit 1
    fi
}

task_lang_map_file="task_lang.json"
model_lang_map_file="llm_lang.json"


# Function to extract languages for a given model
get_model_languages() {
    local model_name="$1"
    local json_file="$2"

    # Use jq to extract languages for the specified model
    jq -r ".llms[] | select(.name == \"$model_name\") | .languages[]" "$task_lang_map_file"
}

# Main processing loops
IFS=',' # Set Internal Field Separator to comma for splitting
for task in "${!taskSubtasksMap[@]}"; do
    subtasks=${taskSubtasksMap[$task]}
    if [[ -z "$subtasks" ]]; then
        # No subtasks defined, fetch all possible ones from tsv
        subtasks_list=("all")
    else
        IFS=',' read -ra subtasks_list <<< "$subtasks"
    fi

    for subtask in "${subtasks_list[@]}"; do
        read -ra languages_list <<< "$languages"

        readarray -t supported_subtask_languages < <(get_task_languages "$task" "$subtask" "$json_file")

        for language in "${languages_list[@]}"; do

            # Check if the language is in the supported languages list
            if [[ ! " ${supported_subtask_languages[*]} " =~ " ${language} " ]]; then
                # If the language is not supported, skip to the next iteration
                echo "Skipping: $language is not supported for $task and $subtask"
                continue
            fi

            read -ra models_list <<< "$models"
            for model in "${models_list[@]}"; do

                readarray -t supported_model_languages < <(get_languages_for_model "$model" "$model_lang_map_file")

                # Check if the language is in the supported by the current model
                if [[ ! " ${supported_model_languages[*]} " =~ " ${language} " ]]; then
                    # If the language is not supported, skip to the next iteration
                    echo "Skipping: $language is not supported for $model"
                    continue
                fi
                echo "Processing Task: $task Subtask: $subtask Language: $language Model: $model"
                # Create a job on the computing cluster
                sbatch noctua2_execute_job.sh $task $subtask $model $language $results_dir
            done
        done
    done
done
