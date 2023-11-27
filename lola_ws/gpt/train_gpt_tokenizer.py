import os
import glob
import json
from tokenizers import ByteLevelBPETokenizer
import random
import datasets
from datasets import load_dataset, Features, Value

data_dir = "/scratch/hpc-prf-lola/data/culturaX/jsonl"
# output_dir = '/scratch/hpc-prf-lola/data/culturaX/tokenizers'
output_dir = '/scratch/hpc-prf-lola/data/culturaX/tokenizers/custom-tokenizer'

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

# # Shuffle function
# def merge_and_shuffle_jsonl(file_paths):
#     """
#     Merge multiple JSONL files and shuffle the rows.

#     Parameters:
#     file_paths (list): List of paths to the JSONL files.
    
#     Returns:
#     list: A list of shuffled items.
#     """
#     all_data = []

#     # Read and collect data from all files
#     for file_path in file_paths:
#         print(f"opening file: {file_path}", flush=True)
#         with open(file_path, 'r', encoding='utf-8') as file:
#             lines = file.readlines()
#             all_data.extend(lines)
#             # for line in lines:
#             #     all_data.append(json.loads(line))
#     print(f"All files read to memory, starting data shuffling...", flush=True)
#     # Shuffle the collected data
#     random.shuffle(all_data)

#     return all_data


# def read_jsonl_files(shuffled_data):
#     batch = []
#     for line in shuffled_data:
#         data = json.loads(line)
#         batch.append(data['text'])
#         if len(batch) == 10000 :
#             yield batch
#             batch = []
#     if len(batch) > 0:
#         yield batch

def filter_empty_files(file_paths):
    """
    Given a list of file paths, divide them into two lists:
    one containing paths of non-empty files, and the other containing paths of empty files.
    """
    non_empty_files = []
    empty_files = []

    for path in file_paths:
        try:
            with open(path, 'r') as file:
                if file.read(1):
                    non_empty_files.append(path)
                else:
                    empty_files.append(path)
        except FileNotFoundError:
            # Consider non-existent files as empty
            empty_files.append(path)

    return non_empty_files, empty_files

# Fetch jsonl paths
culturax_jsonl_paths = get_jsonl_file_paths(data_dir)
#print(culturax_jsonl_paths)
print("Total number of jsonl files: ", len(culturax_jsonl_paths), flush=True)
non_empty_files, empty_files = filter_empty_files(culturax_jsonl_paths)
print("Total number of non-empty jsonl files: ", len(non_empty_files), flush=True)
print("Total number of empty jsonl files: ", len(empty_files), flush=True)
print("Empty jsonl files: ", empty_files, flush=True)
# Create a shuffled object
# shuffled_data = merge_and_shuffle_jsonl(culturax_jsonl_paths)
# for file_path in non_empty_files:
#     print('Loading: ', file_path, '\n\n', flush=True)
#     dataset = load_dataset('json', data_files=file_path, num_proc=128)
# Define the features
features = Features({
    'text': Value('string'),
    'url': Value('string'),
    'timestamp': Value('string'),
    'source': Value('string') 
})
# https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/loading_methods#datasets.load_dataset
#dataset = load_dataset('json', data_files=non_empty_files, features=features, num_proc=128)
datasets.config.IN_MEMORY_MAX_SIZE = 1500000000000 # 1.5 TB
dataset = load_dataset('json', data_files=non_empty_files, features=features, num_proc=128, keep_in_memory=True)
# Performing steps mentioned here: https://huggingface.co/docs/datasets/process#shuffle
iterable_dataset = dataset['train'].to_iterable_dataset(num_shards=128)
# printing status
print("Data loading finished, shuffling the dataset...", flush=True)
shuffled_iterable_dataset = iterable_dataset.shuffle(seed=42, buffer_size=10000)
# Create dataset iterator to return text
def dataset_iterator(dataset, text_field):
    for example in dataset:
        yield example[text_field]
# printing status
print("Data shuffling finished, training tokenizer...", flush=True)
# training tokenizer
tokenizer = ByteLevelBPETokenizer()
# tokenizer.train_from_iterator(read_jsonl_files(shuffled_data), vocab_size=250000, special_tokens=["<pad>", "<eos>", "<unk>", "<s>", "</s>"])
tokenizer.train_from_iterator(dataset_iterator(shuffled_iterable_dataset, 'text'), length=len(dataset['train']), vocab_size=250000, special_tokens=["<pad>", "<eos>", "<unk>", "<s>", "</s>"], show_progress=True)

tokenizer.save_model(output_dir)

print("Done!")
