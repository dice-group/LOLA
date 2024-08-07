#!/bin/bash

## Sample usage: bash restructure_results.sh ../output/lm-eval-harness/

set -eu

# Define the parent directory
parent_dir=$1

# Iterate over each subtask directory in the parent directory
for subtask_dir in "$parent_dir"/*; do
    if [ -d "$subtask_dir" ]; then
        # Iterate over each model-short-name directory in the subtask directory
        for model_short_dir in "$subtask_dir"/*; do
            # Skip if the directory name contains "__"
            if [[ "$model_short_dir" == *"__"* ]]; then
                continue
            fi
            
            # Skip if it's not a directory
            if [ ! -d "$model_short_dir" ]; then
                continue
            fi
            
            # Define the results directory
            results_dir="$model_short_dir/results"
            
            # Skip if there is no results directory
            if [ ! -d "$results_dir" ]; then
                continue
            fi

            # Find the model-full-name directory inside the results directory
            model_full_dir=$(find "$results_dir" -mindepth 1 -maxdepth 1 -type d)

            if [ -d "$model_full_dir" ]; then
                # Move output*.txt to the model-full-name directory
                mv "$model_short_dir"/output*.txt "$model_full_dir"

                # Move the contents of results directory to the subtask directory
                mv "$results_dir"/* "$subtask_dir"

                # Remove the now empty model-short-name directory
                rm -rf "$model_short_dir"
            fi
        done
    fi
done