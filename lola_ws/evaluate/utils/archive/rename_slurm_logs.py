import os
import re
import shutil
import json


def load_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from the file: {file_path}")


model_lang_map_file = "../llm_lang.json"
model_lang_map = load_json_file(model_lang_map_file)

model_id_map = {}

for model in model_lang_map['llms']:
    model_id_map[model['id']] = model['huggingface_model_id'].replace("/", "__")

def extract_variables(filename, pattern):
    match = re.match(pattern, filename)
    if match:
        return match.groups()
    return None

def create_custom_filename(subtask_id, lang, model_id, slurm_id):
    return f"lm-eval-harness_{subtask_id}-{lang}_{model_id}_{slurm_id}.out"

def process_files(src_directory, target_directory, pattern):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        
    for filename in os.listdir(src_directory):
        file_path = os.path.join(src_directory, filename)
        if os.path.isfile(file_path):
            print(f"Processing: {filename}")
            vars = extract_variables(filename, pattern)
            model_id = vars[0]
            # update model id to reflect lm-eval-harness' pattern
            model_id = model_id_map[model_id]
            if vars:
                new_filename = create_custom_filename(vars[1], vars[2], model_id, vars[3])
                new_file_path = os.path.join(target_directory, new_filename)
                shutil.copy(file_path, new_file_path)
                print(f"Copied {filename} to {new_filename}")

# Example usage
src_directory = '../noctua2_logs/old_28-07-2024/'
target_directory = '../noctua2_logs/28-07-2024_v2'
pattern = r"lola-eval-([\w-]+)-lm-eval-harness-([\w-]+)-(\w{2,})_(slurm-\d+)\.out"

process_files(src_directory, target_directory, pattern)