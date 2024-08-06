#!/bin/bash


SEARCH_DIR="/scratch/hpc-prf-lola/data/raw_datasets/CulturaX"
NUM_PROCS=32

PROC_SCRIPT="process_culturax.sh"

mkdir -p proc_logs/

# add languages to exclude if any
## Sometimes we exclude the ones that have already been downloaded, or maybe require too much time to dowload
# declare -a exc_values=("en" "ru" "es" "de" "fr" "zh" "it" "pt")
declare -a exc_values=("none")

#echo "Using for loop with glob pattern:"
for dir in "$SEARCH_DIR"/*; do
    if [ -d "$dir" ]; then
        CUR_LANG=$(basename "$dir")
        # Check for excluded languages
        match_found=false
        for value in "${exc_values[@]}"; do
            if [ "$CUR_LANG" == "$value" ]; then
                match_found=true
                break
            fi
        done
        if [ "$match_found" = false ]; then
            # Note: You may want to increase the time for languages like "en" that are really large, e.g --time=04:00:00
            sbatch -J "processing-culturax-data-${CUR_LANG}" --output="proc_logs/${CUR_LANG}_proc_cultura-slurm_%j.out" $PROC_SCRIPT $CUR_LANG $NUM_PROCS $SEARCH_DIR
        else
            echo "Ignoring ${CUR_LANG} as it is excluded."
        fi
        
    fi
done