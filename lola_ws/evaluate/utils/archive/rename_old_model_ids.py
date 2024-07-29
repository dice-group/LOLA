import os
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
    model_hf_id = model['huggingface_model_id']
    if model_hf_id:
        model_split = model_hf_id.split('/')
        model_id_map[model_split[1]] = model_hf_id.replace("/", "__")


def rename_subsubdirectories(base_directory):
    # Iterate through all subdirectories in the base directory
    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)
        if os.path.isdir(subdir_path):
            # Iterate through all subsubdirectories in each subdirectory
            for subsubdir in os.listdir(subdir_path):
                subsubdir_path = os.path.join(subdir_path, subsubdir)
                if os.path.isdir(subsubdir_path):
                    # Check if the subsubdirectory name is in the custom map
                    if subsubdir in model_id_map:
                        # Get the new name from the custom map
                        new_name = model_id_map[subsubdir]
                        new_subsubdir_path = os.path.join(subdir_path, new_name)
                        # Rename the subsubdirectory
                        os.rename(subsubdir_path, new_subsubdir_path)
                        print(f"Renamed {subsubdir_path} to {new_subsubdir_path}")

# Example usage
base_directory = '../output/lm-eval-harness/'
rename_subsubdirectories(base_directory)