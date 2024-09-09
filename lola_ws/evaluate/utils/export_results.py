import json
import argparse
import os
from tqdm import tqdm
import fnmatch
import pandas as pd
import re
from consts import *

with open('data/model_categories.json', 'r') as category_file:
    model_category_map = json.load(category_file)

# Initialize GROUP_CONSTRAINTS
GROUP_CONSTRAINTS = {}

# Adding KMeans based group contraints
for category_id in set(model_category_map.values()):
    GROUP_CONSTRAINTS[f'category-{category_id}'] = lambda model_info, cid=category_id: model_category_map.get(model_info['id']) == cid

# Adding other group contraints
GROUP_CONSTRAINTS.update({
    'all': lambda model_info: True  # combined group with all models
})

# Adding other group contraints
# GROUP_CONSTRAINTS.update({
#     'lt-4b-params': lambda model_info: model_info['params_in_billions'] <= 2,  # based on Kmeans KVal 2
#     'gt-4b-params': lambda model_info: model_info['params_in_billions'] > 2,   # based on Kmeans KVal 2
# })

# Adding other group contraints
# GROUP_CONSTRAINTS.update({
#     'lt-2b-params': lambda model_info: model_info['params_in_billions'] <= 2,  # Old category
#     'gt-2b-params': lambda model_info: model_info['params_in_billions'] > 2,   # Old category
# })

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
        # fetching the first json object mapped inside the "results" key to the key of the task
        for res_obj in results_info.get("results", {}).items():
            results_dict = res_obj[1]
            break
    return results_dict

def generate_group_map():
    model_info_file_path = "../llm_lang.json"
    model_info_map = load_json_file(model_info_file_path)
    
    model_group_map = { key : set(DEFAULT_MODELS) for key in GROUP_CONSTRAINTS }

    for model_info in model_info_map['llms']:
        model_hf_id = model_info['huggingface_model_id']
        model_params_in_billions = model_info['params_in_billions']
        updated_model_id = model_hf_id.replace("/", "__")
        # find all the groups that this model belongs to
        for group_name, constraint in GROUP_CONSTRAINTS.items():
            if constraint(model_info):
                model_group_map[group_name].add(updated_model_id)
    
    return model_group_map


def export_tsv_results(results_dict, output_dir):
    create_directory_if_not_exists(output_dir)
    model_group_map = generate_group_map()

    # Initialize an array of regex patterns to filter out certain metrics
    metric_filter_patterns = [
        r".*_norm,none.*",  # filter out all normalized metrics
        r".*_stderr.*",  # filter out all metrics with "_stderr"
        r".*alias.*",  # Example: filter out metrics with "alias"
        r".*exact_match,remove_whitespace.*",  # Example: filter out mgsm remove whitespace metric
        r".*exact_match,strict-match.*",  # Example: filter out mgsm strict match metric
        r".*perplexity,none.*",  # Example: filter out lambada perplexity
    ]
    # Unique valid entries count of model, lang & subtask
    unique_val_entries_count = 0
    # Compile regex patterns for efficiency
    compiled_patterns = [re.compile(pattern) for pattern in metric_filter_patterns]

    # Iterate over each subtask in the results dictionary
    for subtask, langs_info in results_dict.items():
        # Create separate DataFrames for each metric and group
        metric_grouped_table_data = { key : {} for key in GROUP_CONSTRAINTS }

        for lang, models_info in langs_info.items():
            for model, metrics in models_info.items():
                # Skip models/subtasks combination that are not valid
                if model in EXCLUDED_MODELS or (subtask, model) in INCOMPATIBLE_SUBTASKS_MODELS:
                    continue

                # Determine the groups of the current model (a model can belong to multiple groups)
                model_groups = []
                for group, models in model_group_map.items():
                    if model in models:
                        model_groups.append(group)

                if not model_groups:
                    continue  # Skip models that do not belong to any group
                # Counting total valid unique entries of model, lang & subtask
                unique_val_entries_count += 1
                # Populate metric_grouped_table_data with data grouped by metrics and model groups
                for metric, value in metrics.items():
                    # Filter out metrics based on the regex patterns
                    if any(pattern.match(metric) for pattern in compiled_patterns):
                        continue  # Skip this metric if it matches any pattern

                    for model_group in model_groups:
                        if metric not in metric_grouped_table_data[model_group]:
                            metric_grouped_table_data[model_group][metric] = pd.DataFrame()

                        # Add the data to the appropriate DataFrame for each group the model belongs to
                        metric_grouped_table_data[model_group][metric].loc[lang, model] = value

        # Export the tables to TSV files, divided by model group
        for group, metrics_data in metric_grouped_table_data.items():
            # Create group directory
            output_group_dir = os.path.join(output_dir, group)
            create_directory_if_not_exists(output_group_dir)
            for metric, df in metrics_data.items():
                # Sort the columns to ensure default models appear first
                sorted_columns = sorted(df.columns, key=lambda x: (x not in DEFAULT_MODELS, x))
                df = df[sorted_columns]

                # Generate the file name based on subtask, metric, and group
                file_name = f"{subtask}_{metric}.tsv"
                file_path = os.path.join(output_group_dir, file_name)

                # Save the DataFrame to a TSV file
                df.to_csv(file_path, sep='\t')
                print(f"Exported {file_name} to {output_dir}")
    
    print(f'Exported total {unique_val_entries_count} unique entries.')
 

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
