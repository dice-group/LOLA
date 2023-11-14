#imports
from datasets import load_dataset
import sys

# The first argument is always the script name.
script_name = sys.argv[0]
# dataset path
DS_PATH=sys.argv[1]
# lang
LANG=sys.argv[2]
# split
SPLIT=sys.argv[3]
# start index
START_IND=sys.argv[4]
# end index
END_IND=sys.argv[5]
# output directory
OUTPUT_DIR=sys.argv[6]


if __name__ == "__main__":
    print(f"Loading dataset for args: {sys.argv[1:]}")
    dataset = load_dataset(DS_PATH, LANG, split=f"{SPLIT}[{START_IND}:{END_IND}]")
    print("Dataset loaded. Starting extraction to jsonl...")
    dataset.to_json(f"{OUTPUT_DIR}/{LANG}_{START_IND}-{END_IND}_ind.jsonl", batch_size=5000, num_proc=32, orient="records", lines=True, force_ascii=False)
    print("jsonl write finished!")