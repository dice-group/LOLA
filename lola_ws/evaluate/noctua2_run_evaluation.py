"""
This script is used to evaluate LLMs on various tasks. The script submits a job to a SLURM-based computing cluster for each unique and valid combination of the provided tasks, models, and languages. This allows for parallel execution and efficient utilization of cluster resources. 
Below is an example of how to run the evaluation with specific models, tasks, and languages.

Command:
    python noctua2_run_evaluation.py --models=XLM-R,mBART --tasks="Mega:XNLI,XCOPA;MultiQ;EleutherAI" --languages=hi,bn --results-dir=./output

Parameters:
    --models: Specifies the models to be used for evaluation. In this example, XLM-R and mBART models are used.
        Example: --models=XLM-R,mBART
    
    --tasks: Specifies the tasks for evaluation, with optional task-specific datasets/sub-tasks. 
        In this example:
            - Mega task includes XNLI and XCOPA datasets.
            - MultiQ task is included without specifying datasets (default datasets will be used).
            - EleutherAI task is included without specifying datasets (default datasets will be used).
        Example: --tasks="Mega:XNLI,XCOPA;MultiQ;EleutherAI"
    
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

NONE_VAL = 'none'
task_lang_map_file = "task_lang.json"
model_lang_map_file = "llm_lang.json"

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

def get_task_languages(task_name, subtask_name, json_file):
    with open(json_file) as f:
        data = json.load(f)

    for task in data['tasks']:
        if task['name'] == task_name:
            if subtask_name and subtask_name != NONE_VAL:
                for subtask in task.get('subtasks', []):
                    if subtask['name'] == subtask_name:
                        return subtask['languages']
            else:
                return task.get('languages', [])
    return []

def get_subtasks_for_task(task_name, json_file):
    with open(json_file) as f:
        data = json.load(f)

    for task in data['tasks']:
        if task['name'] == task_name:
            return [subtask['name'] for subtask in task.get('subtasks', [])] or [NONE_VAL]
    return [NONE_VAL]

def get_model_languages(model_name, json_file):
    with open(json_file) as f:
        data = json.load(f)

    for model in data['llms']:
        if model['name'] == model_name:
            return model['languages']
    return []

def main():
    args = parse_args()

    models = args.models.split(',')
    languages = args.languages.split(',')
    results_dir = args.results_dir

    task_subtask_map = parse_tasks(args.tasks)

    os.makedirs(results_dir, exist_ok=True)

    for task, subtasks in task_subtask_map.items():
        if not subtasks:
            subtasks = get_subtasks_for_task(task, task_lang_map_file)

        for subtask in subtasks:
            supported_subtask_languages = get_task_languages(task, subtask, task_lang_map_file)
            selected_languages = languages
            if 'all' in selected_languages:
                selected_languages = supported_subtask_languages

            for language in selected_languages:
                if language not in supported_subtask_languages:
                    print(f"Skipping: \"{language}\" is not supported for \"{task}\" and \"{subtask}\"")
                    continue

                for model in models:
                    supported_model_languages = get_model_languages(model, model_lang_map_file)

                    if language not in supported_model_languages:
                        print(f"Skipping: \"{language}\" is not supported for model \"{model}\"")
                        continue

                    print(f'Processing Task: "{task}" Subtask: "{subtask}" Language: "{language}" Model: "{model}"')
                    run_name = f"lola-eval-{model}-{task}-{subtask}-{language}"
                    # Create a job on the computing cluster
                    # subprocess.run(['sbatch', '--job-name', run_name, 'noctua2_execute_job.sh', task, subtask, model, language, results_dir])

if __name__ == "__main__":
    main()
