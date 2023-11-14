import json
import os
import subprocess
import sys

### Sample usage: python3 preprocess_large_data.py dist_prep
### Sample usage: python3 preprocess_large_data.py jsonl_prep

script_name = sys.argv[0]

MODE='dist_prep'
if sys.argv[1]:
    MODE=sys.argv[1]


DIST_PROC_LOG_PATH="./dist_prep_logs"
JSONL_PREP_LOG_PATH="./jsonl_prep_logs"

os.makedirs(DIST_PROC_LOG_PATH, exist_ok=True)
os.makedirs(JSONL_PREP_LOG_PATH, exist_ok=True)

STAT_JSON_PATH="./culturax/culturax-v1-0-0_data_stats.json"
CUSTOM_SPLIT_JSON_PATH="./culturax/culturax-custom-data-split.json"

EXCLUDE_LANG_LIST = []
# EXCLUDE_LANG_LIST = ["de", "en", "ru"]
# Read stats json
with open(STAT_JSON_PATH, 'r') as stat_fs:
    stat_json_arr = json.load(stat_fs)
# Read custom split json
with open(CUSTOM_SPLIT_JSON_PATH, 'r') as cus_split_fs:
    cus_split_info = json.load(cus_split_fs)

def_percent=cus_split_info["percent-split"]["default"]
min_doc_limit=cus_split_info["min-doc-count"]

cus_lang_split = {}

for lang_item in stat_json_arr:
    percent_val = def_percent
    lang = lang_item["code"]
    # skip if language is excluded
    if lang in EXCLUDE_LANG_LIST:
        continue
    doc_count = int(lang_item["documents"])
    if lang in cus_split_info["percent-split"]:
        # extract percent val
        percent_val = cus_split_info["percent-split"][lang]
    # Check if lang has atleast minimum number of documents to create a split
    if doc_count < min_doc_limit:
        train_doc_count = doc_count
    else:
        train_doc_count = int(doc_count*percent_val/100)
    # forming dictionary with information about the actual documents to train on for ech language
    cus_lang_split[lang] = train_doc_count

print("Final split information", cus_lang_split)

def submit_data_prepocess_job(lang, start_ind, end_ind):
    # sbatch args
    process_list = ['sbatch']
    process_list.append(f"--job-name={lang}_large_data_prep")
    process_list.append(f"--output={DIST_PROC_LOG_PATH}/%x_slurm_%j.out")
    
    # Script args
    process_list.append(f"preprocess_large_data.sh")
    process_list.append(f"{lang}")
    process_list.append(f"{start_ind}")
    process_list.append(f"{end_ind}")
    
    # run_cmd = f"sbatch --name={lang}_large_data_prep --output={DIST_PROC_LOG_PATH}/%x_slurm_%j.out preprocess_large_data.sh {lang} {start_ind} {end_ind}"
    subprocess.run(process_list)

def submit_json_prep_job(lang, start_ind, end_ind):
    # sbatch args
    process_list = ['sbatch']
    process_list.append(f"--job-name={lang}_jsonl_prep")
    process_list.append(f"--output={JSONL_PREP_LOG_PATH}/%x_slurm_%j.out")
    
    # Script args
    process_list.append(f"prepare_jsonl.sh")
    process_list.append(f"{lang}")
    process_list.append(f"{start_ind}")
    process_list.append(f"{end_ind}")
    
    subprocess.run(process_list)

prep_fn = submit_data_prepocess_job
if MODE == 'jsonl_prep':
    prep_fn = submit_json_prep_job

# For each language
for lang in cus_lang_split:
    start_ind = 0
    end_ind = cus_lang_split[lang] - 1

    prep_fn(lang, start_ind, end_ind)

print(f"All jobs submitted! Check {DIST_PROC_LOG_PATH if MODE=='dist_prep' else JSONL_PREP_LOG_PATH} for slurm logs.")
    

    