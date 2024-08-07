import json
import argparse
import os
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_args():
    parser = argparse.ArgumentParser(description="Find the combination of model and language for which a task failed to execute.")
    parser.add_argument('--main_task_id', required=True, help="Id of the main task.")
    parser.add_argument('--root_results_dir', default="../output", help="Root directory of the experiment results (default: ../output).")
    parser.add_argument('--root_logs_dir', default="../noctua2_logs", help="Root directory of the experiment logs (default: ../noctua2_logs).")
    parser.add_argument('--logs_sub_dirs', default="all", help="Comma separated list of subdirectories, put \"all\" if all sub directories are to be used.")
    parser.add_argument('--output_summary_file', default="./error_summary.txt", help="Filepath to write error summary into.")

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
    # The patterns below should be extended if required. The order of the patterns in the list is important. More patterns will make the processing slower.
    error_patterns = [
        re.compile(r'.*(torch\.OutOfMemoryError: CUDA out of memory\.).*'),
        re.compile(r'.*(RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED).*'),
        re.compile(r'.*(FileNotFoundError: No such file or directory).*'),
        re.compile(r'(.*AssertionError.*)'),
        re.compile(r'(.*Exception: .+)'),
        re.compile(r'(.*Error: .+)')
    ]
    errors = []
    try:
        with open(log_file_path, 'r') as log_file:
            for line in log_file:
                for pattern in error_patterns:
                    match = pattern.search(line)
                    if match:
                        errors.append(match.group(1))
                        # break at the first error
                        break
    except FileNotFoundError:
        raise FileNotFoundError(f"Log file not found: {log_file_path}")
    
    return errors

def process_log_files(log_files):
    error_summary = {}
    for log_file in log_files:
        errors = extract_error_from_log(log_file)
        if not errors:
            print(f"Couldn't extract errors from: {log_file}")
        for error in errors:
            if error not in error_summary:
                error_summary[error] = []
            error_summary[error].append(log_file)
    return error_summary

def main():
    # parse input arguments
    args = parse_args()
    task_lang_map_file = "../task_lang.json"
    task_lang_map = load_json_file(task_lang_map_file)
    main_task_id = args.main_task_id
    # verify the task name
    main_task_map = next((task for task in task_lang_map['tasks'] if task['id'] == main_task_id), None)
    if not main_task_map:
        raise ValueError(f"No information found for the provided task id: {main_task_id}")
    
    root_results_dir = os.path.abspath(args.root_results_dir)
    main_task_res_dir = os.path.join(root_results_dir, main_task_id)
    # verify the results directory
    if not os.path.isdir(main_task_res_dir):
        raise FileNotFoundError(f"Path not found: {main_task_res_dir}")
    
    print(f'Scanning the subtasks for errors: {main_task_res_dir}')

    missing_combinations = {}
    total_missing_results = 0
    # find entries with missing results
    for subtask in tqdm(os.listdir(main_task_res_dir)):
        subtask_dir = os.path.join(main_task_res_dir, subtask)
        
        if os.path.isdir(subtask_dir):
            for model in os.listdir(subtask_dir):
                model_dir = os.path.join(subtask_dir, model)
                if os.path.isdir(model_dir):
                    if not find_results_json(model_dir):
                        if subtask not in missing_combinations:
                            missing_combinations[subtask] = []
                        missing_combinations[subtask].append(model)
                        total_missing_results += 1
    # find the relevant log files
    if missing_combinations:
        print(f" Total {total_missing_results} missing results.")
        root_logs_dir = os.path.abspath(args.root_logs_dir)
        logs_sub_dirs = args.logs_sub_dirs.split(",")
        
        if "all" in logs_sub_dirs:
            logs_sub_dirs = os.listdir(root_logs_dir)
        print(f'Extracting errors from the log files: {root_logs_dir}')
        
        log_files_to_process = []

        for subtask in missing_combinations:
            model_list = missing_combinations[subtask]
            for model in model_list:
                for log_sub_dir in logs_sub_dirs:
                    sub_dir_path = os.path.join(root_logs_dir, log_sub_dir)
                    if os.path.isdir(sub_dir_path):
                        for root, _, files in os.walk(sub_dir_path):
                            for file in files:
                                run_name = f"{main_task_id}_{subtask}_{model}"
                                log_file_pattern = run_name + r'_slurm-\d+\.out'
                                if re.match(log_file_pattern, file):
                                    log_files_to_process.append((subtask, model, os.path.join(root, file)))
        
        error_summary = {}
        total_extracted_errors = 0
        # extract errors
        with ThreadPoolExecutor() as executor:
            future_to_log = {executor.submit(extract_error_from_log, log_file[2]): log_file for log_file in log_files_to_process}
            for future in tqdm(as_completed(future_to_log), total=len(future_to_log)):
                log_file = future_to_log[future]
                try:
                    errors = future.result()
                    for error in errors:
                        if error not in error_summary:
                            error_summary[error] = {}
                        if log_file[0] not in error_summary[error]:
                            error_summary[error][log_file[0]] = []
                        error_summary[error][log_file[0]].append(log_file[1])
                        total_extracted_errors += 1
                except Exception as exc:
                    print(f'{log_file} generated an exception: {exc}')
        # Write the error summary
        output_summary_file = args.output_summary_file
        with open(output_summary_file, 'w') as sum_out:
            sum_out.write(f"Total {total_extracted_errors}/{total_missing_results} errors found.")
            for error, tasks_models in error_summary.items():
                local_error_count = 0
                for subtask, models in tasks_models.items():
                    local_error_count += len(models)
                sum_out.write(f"\nError encountered {local_error_count} time(s): {error}")
                sum_out.write(f"\t{tasks_models}")
                
            print(f"Error summary exported to: {output_summary_file}")
    else:
        print("No missing combinations found.")
    
if __name__ == "__main__":
    main()