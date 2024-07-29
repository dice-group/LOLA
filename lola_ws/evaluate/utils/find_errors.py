import json
import argparse
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Find the combination of model and language for which a task failed to execute.")
    parser.add_argument('--main_task_id', required=True, help="Id of the main task.")
    parser.add_argument('--results-dir', default="./results", help="Directory where the experiment results can be found (default: ./results)")

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
    results_dir = os.path.abspath(args.results_dir)
    main_task_res_dir = os.path.join(results_dir, main_task_id)
    
    # Check if the results path exists
    if not os.path.isdir(main_task_res_dir):
        raise FileNotFoundError(f"Path not found: {main_task_res_dir}")
    
    print(f'Scanning the subtasks for errors: {main_task_res_dir}')

    missing_combinations = []

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
        print(f" Total {len(missing_combinations)}  missing results:")
        for subtask, model in missing_combinations:
            print(f"Subtask: {subtask}, Model: {model}")
    else:
        print("No missing combinations found.")
    

if __name__ == "__main__":
    main()