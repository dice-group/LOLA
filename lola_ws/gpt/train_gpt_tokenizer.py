import os
import glob
import json
from tokenizers import ByteLevelBPETokenizer
import random

data_dir = "/scratch/hpc-prf-lola/data/culturaX/jsonl"
output_dir = '/scratch/hpc-prf-lola/data/culturaX/tokenizers'

def get_jsonl_file_paths(directory):
    """
    Get a list of paths to all JSONL files in the specified directory.

    Parameters:
    directory (str): The path to the directory containing JSONL files.

    Returns:
    list: A list of paths to the JSONL files.
    """
    # Construct the pattern to match JSONL files
    pattern = os.path.join(directory, '*.jsonl')

    # Use glob to find all files matching the pattern
    file_paths = glob.glob(pattern)

    return file_paths

# Shuffle function
def merge_and_shuffle_jsonl(file_paths):
    """
    Merge multiple JSONL files and shuffle the rows.

    Parameters:
    file_paths (list): List of paths to the JSONL files.
    
    Returns:
    list: A list of shuffled items.
    """
    all_data = []

    # Read and collect data from all files
    for file_path in file_paths:
        print(f"opening file: {file_path}", flush=True)
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            all_data.extend(lines)
            # for line in lines:
            #     all_data.append(json.loads(line))
    print(f"All files read to memory, starting data shuffling...", flush=True)
    # Shuffle the collected data
    random.shuffle(all_data)

    return all_data


def read_jsonl_files(shuffled_data):
    batch = []
    for line in shuffled_data:
        data = json.loads(line)
        batch.append(data['text'])
        if len(batch) == 10000 :
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


# Fetch jsonl paths
culturax_jsonl_paths = get_jsonl_file_paths(data_dir)
#print(culturax_jsonl_paths)
print("Total number of jsonl files: ", len(culturax_jsonl_paths), flush=True)
# Create a shuffled object
shuffled_data = merge_and_shuffle_jsonl(culturax_jsonl_paths)
# training tokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(read_jsonl_files(shuffled_data), vocab_size=250000, special_tokens=["<pad>", "<eos>", "<unk>", "<s>", "</s>"])

tokenizer.save_pretrained(output_dir)

print("Done!")
