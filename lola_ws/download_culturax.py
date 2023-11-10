import os
import glob
from datasets import load_dataset, DownloadConfig

NUM_PROCS=128

lang_dir = '/scratch/hpc-prf-lola/nikit/.cache/huggingface/datasets--uonlp--CulturaX/snapshots/321a983f3fd2a929cc1f8ef6207834bab0bb9e25'

languages = [f for f in os.listdir(lang_dir) if os.path.isdir(os.path.join(lang_dir, f))]
print(languages)

download_config = DownloadConfig(
    resume_download=True,  # You can choose 'reuse_dataset_if_exists' to avoid redownloading
    num_proc=NUM_PROCS,
    max_retries=10
)

for lang in languages:
    try:
        #search_pattern = os.path.join(f"{lang_dir}/{lang}", "*.parquet")
        #parquet_files = glob.glob(search_pattern)
        #print(parquet_files)
        print('Downloading for language: ', lang)
        load_dataset("uonlp/CulturaX", language=lang, split='train', num_proc=NUM_PROCS,  download_config=download_config)
    except Exception as e:
        print("Exception: ",e)