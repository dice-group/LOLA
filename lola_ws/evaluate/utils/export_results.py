import json
import argparse
import os
from tqdm import tqdm
import fnmatch
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Compile all the evaluation results and export them to tsv files.")
    parser.add_argument('--main_task_id', required=True, help="Id of the main task.")
    parser.add_argument('--root_results_dir', default="../output", help="Root directory of the experiment results (default: ../output).")
    parser.add_argument('--export_directory', default="../evaluation_tables", help="Directory to which export the result tsv tables to  (default: ../evaluation_tables).")

    return parser.parse_args()

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from the file: {file_path}")

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def find_results_file(directory):
    # Ensure the directory exists
    if not os.path.isdir(directory):
        return None

    # List all files in the directory
    files = os.listdir(directory)

    # Filter files that match the pattern 'results*.json'
    matching_files = fnmatch.filter(files, 'results*.json')

    # Return the first matching file if any, else return None
    return matching_files[0] if matching_files else None

def fetch_result_dict(results_dir):
    results_file = find_results_file(results_dir)
    results_dict = None
    if results_file:
        results_info = load_json_file(os.path.join(results_dir, results_file))
        #print(results_info.get("results", {}).items())
        # fetching the first json object mapped inside the "results" key to the key of the task
        for res_obj in results_info.get("results", {}).items():
            results_dict = res_obj[1]
            break
    return results_dict

def export_tsv_results(results_dict, output_dir):
    create_directory_if_not_exists(output_dir)
    for subtask, langs_info in results_dict.items():
        # Create an empty dataframe
        df = pd.DataFrame()

        for lang, models_info in langs_info.items():
            row_data = {}
            for model, metrics in models_info.items():
                for metric, value in metrics.items():
                    column_name = f"{model}_{metric}"
                    row_data[column_name] = value
            df = df.append(pd.DataFrame(row_data, index=[lang]))

        # Save to TSV
        df.to_csv(f"{output_dir}/{subtask}.tsv", sep='\t')
    
def main():
     # parse input arguments
    args = parse_args()
    # name of the task
    main_task_id = args.main_task_id

    task_lang_map_file = "../task_lang.json"
    task_lang_map = load_json_file(task_lang_map_file)
    
    # verify the task name
    main_task_list = next((task for task in task_lang_map['tasks'] if task['id'] == main_task_id), None)
    if not main_task_list:
        raise ValueError(f"No information found for the provided task id: {main_task_id}")
    
    root_results_dir = os.path.abspath(args.root_results_dir)
    main_task_res_dir = os.path.join(root_results_dir, main_task_id)
    # export map
    export_result_map = {}
    # Iterate through all the subtasks
    for subtask_info in tqdm(main_task_list['subtasks']):
        #print(subtask_info)
        # generate directory names, based on formatted_id for all supported langauges
        formatted_id = subtask_info.get('formatted_id', None)
        subtask_id = subtask_info.get('id', None)
        export_result_map[subtask_id] = {}
        # loop through the directories
        for sup_lang in subtask_info['languages']:
            export_result_map[subtask_id][sup_lang] = {}
            subtask_dir_name = formatted_id % sup_lang if formatted_id else subtask_id + '_' + sup_lang
            subtask_dir = os.path.join(main_task_res_dir, subtask_dir_name)
            # loop through the models
            if os.path.isdir(subtask_dir):
                for model in os.listdir(subtask_dir):
                    model_dir = os.path.join(subtask_dir, model)
                    if os.path.isdir(model_dir):
                        # if the result file is present
                        model_subtask_result = fetch_result_dict(model_dir)
                        if model_subtask_result:
                            # extract and map the result
                            export_result_map[subtask_id][sup_lang][model] = model_subtask_result

    # Export the map as tsv   
    print('Exporting result tables..')
    export_tsv_results(export_result_map, args.export_directory)

if __name__ == "__main__":
    main()