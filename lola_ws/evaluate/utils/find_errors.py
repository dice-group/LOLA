import json
import argparse
import os
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from consts import *

# experiment command template
EXP_CMD_TMPLT= 'python3 noctua2_run_evaluation.py --models="%s" --tasks="%s" --languages="%s" --results_dir="%s"'
# Loading task and llm mappings
task_lang_map_file = "../task_lang.json"
model_lang_map_file = "../llm_lang.json"

with open(task_lang_map_file) as f1, open(model_lang_map_file) as f2:
    task_lang_map = json.load(f1)
    model_lang_map = json.load(f2)

def parse_args():
    parser = argparse.ArgumentParser(description="Find the combination of model and language for which a task failed to execute.")
    parser.add_argument('--main_task_id', required=True, help="Id of the main task.")
    parser.add_argument('--root_results_dir', default="../output", help="Root directory of the experiment results (default: ../output).")
    parser.add_argument('--root_logs_dir', default="../noctua2_logs", help="Root directory of the experiment logs (default: ../noctua2_logs).")
    parser.add_argument('--logs_sub_dirs', default="all", help="Comma separated list of subdirectories, put \"all\" if all sub directories are to be used.")
    parser.add_argument('--output_summary_file', default="./error_summary.txt", help="Filepath to write error summary into.")
    parser.add_argument('--output_rerun_file', default="./rerun_cmds.txt", help="Filepath to write rerun commands into.")

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
        re.compile(r'.*(slurmstepd: error:.*\*\*\*.+\*\*\*)'),
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
                else:
                    continue  # only executed if the inner loop did NOT break
                break  # only executed if the inner loop DID break
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

def export_rerun_commands(commands_to_rerun, rerun_cmd_file):
    try:
        with open(rerun_cmd_file, 'w') as file:
            for item in commands_to_rerun:
                file.write(f"{item}\n")
        print(f"Rerun commands successfully exported to {rerun_cmd_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


def gen_rerun_command(model_name, main_task_id, subtask_name, result_path):
    # Convert the model name back to normal and find the respective model id from llm_lang.json
    model_hf_id = model_name.replace("__", "/")
    model_id = None
    final_subtask_id = None
    extracted_lang = None
    exe_cmd = None
    for model in model_lang_map['llms']:
        if model['huggingface_model_id'] == model_hf_id:
            model_id = model['id']
            break
    # match subtask name with all task formatted_id or id to extract the subtask id and language
    for task in task_lang_map['tasks']:
        if task['id'] == main_task_id:
            for subtask in task['subtasks']:
                subtask_id = subtask['id']
                formatted_id = subtask.get('formatted_id', None)
                # FIXME: the if clause below is very specific and needs a logic update in future.
                if all(sub in subtask_name for sub in subtask_id.split('_')):
                    final_subtask_id = subtask_id
                    if formatted_id:
                        # extract lang out of formatted id
                        subtask_regex = re.escape(formatted_id).replace("%s", "(.+)")
                    else:
                        # extract lang based on subtask id regex
                        subtask_regex = subtask_id + r'_(.+)'
                    # Use regex to find the match
                    match = re.match(subtask_regex, subtask_name)
                    if match:
                        extracted_lang = match.group(1)
                    # break because we found the matching subtask, no point in looking further.
                    break
            # break because we found the matching task, no point in looking further.
            break
    
    # generate the evaluation execution command
    if model_id and final_subtask_id and extracted_lang and result_path:
        exe_cmd = EXP_CMD_TMPLT % (model_id, main_task_id + ':' + final_subtask_id, extracted_lang, result_path)
    # return
    if not exe_cmd:
        print('Couldn\'t generate command for: ', model_name, main_task_id, subtask_name, result_path)
        print('Extracted values: ', model_id , final_subtask_id , extracted_lang , result_path)
        print(f'Subtask regex: {subtask_regex}')
    return exe_cmd

def main():
    # parse input arguments
    args = parse_args()
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

    total_missing_results = 0

    # find the relevant log files
    root_logs_dir = os.path.abspath(args.root_logs_dir)
    logs_sub_dirs = args.logs_sub_dirs.split(",")
    
    if "all" in logs_sub_dirs:
        logs_sub_dirs = os.listdir(root_logs_dir)
    print(f'Extracting errors from the log files: {root_logs_dir}')
    
    log_files_to_process = []
    commands_to_rerun = []

    # Note: be careful before touching the regex below, we considered the constraints for huggingface organization and model to accurately extract the groups.
    log_file_pattern = re.compile(
        rf'{re.escape(main_task_id)}_(.+?)_([A-Za-z0-9-]+__.+?)_slurm-\d+\.out'
    )
    # Maintaining this set to check for duplicates
    failed_subtask_model_set = set()
    all_subtask_model_set = set()
    for log_sub_dir in logs_sub_dirs:
        sub_dir_path = os.path.join(root_logs_dir, log_sub_dir)
        if os.path.isdir(sub_dir_path):
            for root, _, files in os.walk(sub_dir_path):
                print(f'Processing: {sub_dir_path}')
                for file in tqdm(files):
                    match = log_file_pattern.match(file)
                    if match:
                        subtask = match.group(1)
                        model = match.group(2)
                        result_subdir = os.path.join(main_task_res_dir, subtask, model)
                        
                        # Ignore if the model is excluded
                        if model in EXCLUDED_MODELS:
                            continue
                        # Record the combination
                        subtask_model_comb = subtask + '+' + model
                        all_subtask_model_set.add(subtask_model_comb)
                        if not os.path.exists(result_subdir) or not find_results_json(result_subdir):
                            # check if this combination has already been found before
                            if subtask_model_comb in failed_subtask_model_set:
                                # skip the duplicate error
                                continue
                            failed_subtask_model_set.add(subtask_model_comb)
                            total_missing_results += 1
                            # Generate the experiment rerun command
                            gen_cmd = gen_rerun_command(model, main_task_id, subtask, root_results_dir)
                            commands_to_rerun.append(gen_cmd)
                            log_files_to_process.append((subtask, model, os.path.join(root, file)))

    error_summary = {}
    total_extracted_errors = 0
    total_unique_experiments = len(all_subtask_model_set)
    # extract errors
    print('Extracting errors')
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
        first_line = f"Total {total_extracted_errors}/{total_missing_results} errors found for {total_unique_experiments} unique experiments."
        print(first_line)
        sum_out.write(first_line)
        for error, tasks_models in error_summary.items():
            local_error_count = 0
            for subtask, models in tasks_models.items():
                local_error_count += len(models)
            sum_out.write(f"\nError encountered {local_error_count} time(s): {error}")
            sum_out.write(f"\t{tasks_models}")
     
    print(f"Error summary exported to: {output_summary_file}")
    
    export_rerun_commands(commands_to_rerun, args.output_rerun_file)
    
if __name__ == "__main__":
    main()
