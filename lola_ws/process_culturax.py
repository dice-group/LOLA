import os
import glob
from datasets import load_dataset, DownloadConfig

import sys

# The first argument is always the script name.
script_name = sys.argv[0]

CUR_LANG=sys.argv[1]
NUM_PROCS=int(sys.argv[2])
SEARCH_DIR=sys.argv[3]
# CUR_LANG="en"
# NUM_PROCS=64
# SEARCH_DIR="/scratch/hpc-prf-lola/data/raw_datasets/CulturaX"


# lang_dir = '/scratch/hpc-prf-lola/nikit/.cache/huggingface/datasets--uonlp--CulturaX/snapshots/321a983f3fd2a929cc1f8ef6207834bab0bb9e25'

# languages = [f for f in os.listdir(lang_dir) if os.path.isdir(os.path.join(lang_dir, f))]
# print(languages)

# download_config = DownloadConfig(
#     resume_download=True,  # You can choose 'reuse_dataset_if_exists' to avoid redownloading
#     num_proc=32,
#     max_retries=10
# )

# for lang in languages:
#     try:
#         #search_pattern = os.path.join(f"{lang_dir}/{lang}", "*.parquet")
#         #parquet_files = glob.glob(search_pattern)
#         #print(parquet_files)
#         print('Downloading for language: ', lang)
#         load_dataset("uonlp/CulturaX", language=lang, split='train', num_proc=NUM_PROCS,  download_config=download_config)
#     except Exception as e:
#         print("Exception: ",e)

# It will generate the splits (arrow files) if not yet done
load_dataset(SEARCH_DIR, CUR_LANG, num_proc=NUM_PROCS)

print(f"Processing finished for {CUR_LANG}")