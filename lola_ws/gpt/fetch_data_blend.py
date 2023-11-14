# script.py
import sys
import os
import re
import json

STAT_JSON_PATH="./culturax/culturax-v1-0-0_data_stats.json"
EXCLUDE_LANG_LIST = []

with open(STAT_JSON_PATH, 'r') as stat_fs:
    stat_json_arr = json.load(stat_fs)

def find_directory(target_dir, regex_pattern):
    # Check if the target directory exists
    if not os.path.exists(target_dir):
        return None

    # Compile the regular expression pattern
    pattern = re.compile(regex_pattern)

    # Iterate through the items in the target directory
    with os.scandir(target_dir) as entries:
        for entry in entries:
            if entry.is_dir() and pattern.match(entry.name):
                # A matching directory is found
                return entry.path

    # No matching directory found
    return None

def compose_data_blend_string(data_root_path):
    data_blend_string = ""
    # For each language
    for lang_item in stat_json_arr:
        lang = lang_item["code"]
        # skip if language is excluded
        if lang in EXCLUDE_LANG_LIST:
            continue
        # Find the directory
        lang_dir = find_directory(data_root_path, r'data-'+ re.escape(lang) +r'-\d+-\d+-\w+')
        if lang_dir:
            blend_item=f"1 {lang_dir}/meg-culturax-{lang}_text_document "
            data_blend_string+= blend_item
    return data_blend_string
        

if __name__ == "__main__":
    # Pass all command line arguments except the script name
    data_blend_string = compose_data_blend_string(sys.argv[1])
    print(data_blend_string)