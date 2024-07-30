import os
import re

# Define the pattern
pattern = re.compile(r'lm-eval-harness_truthfulqa_mc1_(.{2,})_(.+__.+_slurm-\d+\.out)')

# Custom function to generate new file names
def custom_file_name(group1, group2):
    # Implement your custom naming logic here
    # For example, let's concatenate the groups with some custom text
    return f"lm-eval-harness_truthfulqa_{group1}_mc1_{group2}"

# Directory containing the files
directory = '/scratch/hpc-prf-lola/nikit/repos/LOLA-Evaluate/lola_ws/evaluate/noctua2_logs/28-07-2024'

# Iterate over all files in the directory
for filename in os.listdir(directory):
    match = pattern.match(filename)
    if match:
        group1, group2 = match.groups()
        new_filename = custom_file_name(group1, group2)
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        os.rename(old_filepath, new_filepath)
        print(f'Renamed: {filename} -> {new_filename}')

print('Renaming completed.')