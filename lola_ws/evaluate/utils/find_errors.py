import json
import argparse
import os
import re
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Find the combination of model and language for which a task failed to execute.")
    parser.add_argument('--main_task_id', required=True, help="Id of the main task.")
    parser.add_argument('--root_results_dir', default="../output", help="Root directory of the experiment results (default: ../output).")
    parser.add_argument('--root_logs_dir', default="../noctua2_logs", help="Root directory of the experiment logs (default: ../noctua2_logs).")
    parser.add_argument('--logs_sub_dirs', default="all", help="Comma separated list of subdirectories, put \"all\" if all sub directories are to be used.")

    return parser.parse_args()

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from the file: {file_path}")

def find_results_json(directory):
    """Recursively search for results_*.json files within the given directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith('results_') and file.endswith('.json'):
                return True
    return False

def extract_error_from_log(log_file_path):
    """Extract error/exception messages from the log file."""
    # TODO: requires refinement to extract all types of errors and exceptions
    error_patterns = [
        re.compile(r'.*Exception: (.+)'),
        re.compile(r'.*Error: (.+)')
    ]
    errors = []
    try:
        with open(log_file_path, 'r') as log_file:
            for line in log_file:
                for pattern in error_patterns:
                    match = pattern.search(line)
                    if match:
                        errors.append(match.group(1))
    except FileNotFoundError:
        raise FileNotFoundError(f"Log file not found: {log_file_path}")
    
    return errors

def replace_last(string, old, new):
    return new.join(string.rsplit(old, 1))

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Load task mappings
    task_lang_map_file = "../task_lang.json"
    #model_lang_map_file = "../llm_lang.json"
    
    task_lang_map = load_json_file(task_lang_map_file)
    #model_lang_map = load_json_file(model_lang_map_file)
    
    # main task id
    main_task_id = args.main_task_id
    
    # Verify main task id
    main_task_map = next((task for task in task_lang_map['tasks'] if task['id'] == main_task_id), None)
    if not main_task_map:
        raise ValueError(f"No information found for the provided task id: {main_task_id}")
    
    # Process results directory
    root_results_dir = os.path.abspath(args.root_results_dir)
    main_task_res_dir = os.path.join(root_results_dir, main_task_id)
    
    # Check if the results path exists
    if not os.path.isdir(main_task_res_dir):
        raise FileNotFoundError(f"Path not found: {main_task_res_dir}")
    
    print(f'Scanning the subtasks for errors: {main_task_res_dir}')

    missing_combinations = []
    error_summary = {}

    # Iterate over subtasks
    for subtask in tqdm(os.listdir(main_task_res_dir)):
        subtask_dir = os.path.join(main_task_res_dir, subtask)
        
        if os.path.isdir(subtask_dir):
            # Iterate over models
            for model in os.listdir(subtask_dir):
                model_dir = os.path.join(subtask_dir, model)
                if os.path.isdir(model_dir):
                    # Check for nested JSON files in the format results_*.json
                    if not find_results_json(model_dir):
                        missing_combinations.append((subtask, model))
    
    if missing_combinations:
        print(f" Total {len(missing_combinations)} missing results:")
        for subtask, model in missing_combinations:
            print(f"Subtask: {subtask}, Model: {model}")
            
            # Locate the log files for the specific subtask and model
            root_logs_dir = os.path.abspath(args.root_logs_dir)
            logs_sub_dirs = args.logs_sub_dirs.split(",")
            
            if "all" in logs_sub_dirs:
                logs_sub_dirs = os.listdir(root_logs_dir)
            
            log_files = []
            for log_sub_dir in logs_sub_dirs:
                sub_dir_path = os.path.join(root_logs_dir, log_sub_dir)
                if os.path.isdir(sub_dir_path):
                    for root, _, files in os.walk(sub_dir_path):
                        for file in files:
                            # File should match the pattern combining main_task_id, subtask & model
                            run_name = f"{main_task_id}_{replace_last(subtask, '_', '-')}_{model}"
                            log_file_pattern = run_name + r'_slurm-\d+\.out'
                            if re.match(log_file_pattern, file):
                                log_files.append(os.path.join(root, file))
            print(log_files)
            # Extract errors from log files
            for log_file in log_files:
                errors = extract_error_from_log(log_file)
                for error in errors:
                    if error not in error_summary:
                        error_summary[error] = []
                    error_summary[error].append((subtask, model))
        
        # Print the error summary
        print("\nError Summary:")
        for error, tasks_models in error_summary.items():
            print(f"\nError: {error}")
            for subtask, model in tasks_models:
                print(f"  Subtask: {subtask}, Model: {model}")
    else:
        print("No missing combinations found.")
    

if __name__ == "__main__":
    main()