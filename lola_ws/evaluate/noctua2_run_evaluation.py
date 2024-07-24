"""
This script is used to evaluate LLMs on various tasks. The script submits a job to a SLURM-based computing cluster for each unique and valid combination of the provided tasks, models, and languages. This allows for parallel execution and efficient utilization of cluster resources. Below is an example of how to run the evaluation with specific models, tasks, and languages.

Command:
    python3 noctua2_run_evaluation.py --models=xlmr,mbart --tasks="lm-eval-harness:lm-eval-harness_xcopa,lm-eval-harness_xnli;okapi" --languages=hi,bn --results-dir=./output

Parameters:
    --models: Specifies the models to be used for evaluation. In this example, xlmr and mbart models are used.
        Example: --models=xlmr,mbart
    
    --tasks: Specifies the tasks for evaluation, with optional task-specific datasets/sub-tasks. 
        In this example:
            - lm-eval-harness task includes lm-eval-harness_xcopa and lm-eval-harness_xnli datasets.
            - okapi task is included without specifying datasets (default datasets will be used).
        Example: --tasks="lm-eval-harness:lm-eval-harness_xcopa,lm-eval-harness_xnli;okapi"
    
    --languages: Specifies the languages to evaluate the models on. In this example, Hindi (hi) and Bengali (bn) are used.
        Example: --languages=hi,bn
    
    --results-dir: Specifies the directory where the evaluation results will be saved.
        Example: --results-dir=./output

To run this evaluation, simply copy the command above and execute it in your terminal.
"""


import argparse
import os
import json
import subprocess
from datetime import datetime

# Get today's date
today = datetime.today()
# Format the date as dd-mm-yyyy
formatted_date = today.strftime('%d-%m-%Y')

# Creating directory for SLURM logs
def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

slurm_log_path = 'noctua2_logs/' + formatted_date + '/'
create_directory_if_not_exists(slurm_log_path)

NONE_VAL = 'none'
task_lang_map_file = "task_lang.json"
model_lang_map_file = "llm_lang.json"

with open(task_lang_map_file) as f1, open(model_lang_map_file) as f2:
    task_lang_map = json.load(f1)
    model_lang_map = json.load(f2)


def parse_args():
    parser = argparse.ArgumentParser(description="Process some models, tasks, and languages.")
    parser.add_argument('--models', required=True, help="Comma-separated list of models (e.g., 'model1,model2')")
    parser.add_argument('--tasks', required=True, help="Tasks with optional subtasks formatted as 'task1:sub1,sub2;task2'")
    parser.add_argument('--languages', required=True, help="Comma-separated list of languages (e.g., 'en,es')")
    parser.add_argument('--results-dir', default="./results", help="Optional directory for saving results (default: ./results)")

    return parser.parse_args()

def parse_tasks(tasks):
    task_subtask_map = {}
    task_list = tasks.split(';')
    for item in task_list:
        task, *subtasks = item.split(':')
        task_subtask_map[task] = subtasks[0].split(',') if subtasks else []
    return task_subtask_map

def get_task_languages(task_id, subtask_id):
    data = task_lang_map

    for task in data['tasks']:
        if task['id'] == task_id:
            if subtask_id and subtask_id != NONE_VAL:
                for subtask in task.get('subtasks', []):
                    if subtask['id'] == subtask_id:
                        return subtask['languages']
            else:
                return task.get('languages', [])
    return []

def get_subtasks_for_task(task_id):
    data = task_lang_map

    for task in data['tasks']:
        if task['id'] == task_id:
            return [subtask['id'] for subtask in task.get('subtasks', [])] or [NONE_VAL]
    return [NONE_VAL]

def get_model_information(model_id):
    data = model_lang_map

    for model in data['llms']:
        if model['id'] == model_id:
            return model['huggingface_model_id'], model['languages']
    return []

def main():
    args = parse_args()

    models = args.models.split(',')
    languages = args.languages.split(',')
    results_dir = args.results_dir
    results_dir = os.path.abspath(results_dir)

    task_subtask_map = parse_tasks(args.tasks)

    # os.makedirs(results_dir, exist_ok=True)
    create_directory_if_not_exists(results_dir)

    for task, subtasks in task_subtask_map.items():
        if not subtasks:
            subtasks = get_subtasks_for_task(task)

        for subtask in subtasks:
            supported_subtask_languages = get_task_languages(task, subtask)
            selected_languages = languages
            if 'all' in selected_languages:
                selected_languages = supported_subtask_languages

            for language in selected_languages:
                if language not in supported_subtask_languages:
                    print(f"Skipping: \"{language}\" is not supported for \"{task}\" and \"{subtask}\"")
                    continue

                for model in models:
                    model_hf_id, supported_model_languages = get_model_information(model)

                    if language not in supported_model_languages:
                        print(f"Skipping: \"{language}\" is not supported for model \"{model}\"")
                        continue

                    print(f'Processing Task: "{task}" Subtask: "{subtask}" Language: "{language}" Model: "{model}" Huggingface ID: "{model_hf_id}"')
                    run_name = f"lola-eval-{model}-{task}-{subtask}-{language}"
                    # Create a job on the computing cluster
                    sub_proc_arr = ['sbatch', '--job-name', run_name, '--output', (slurm_log_path + '%x_slurm-%j.out'), 'noctua2_execute_job.sh', task, subtask, model_hf_id, language, results_dir]
                    print("Subprocess called: ", sub_proc_arr)
                    subprocess.run(sub_proc_arr)

if __name__ == "__main__":
    main()
