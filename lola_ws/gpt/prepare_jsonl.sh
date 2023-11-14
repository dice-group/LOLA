#!/bin/bash
#SBATCH -t 06:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu 2G

# Sample usage: sbatch prepare_jsonl.sh gu 0 21144

set -eu

CUR_LANG=$1
START_IND=$2
END_IND=$3

export VENV_PATH=~/virt-envs/venv-lola
export OUTPUT_PATH=/scratch/hpc-prf-lola/data/culturaX/jsonl/mgpt-tokenized

mkdir -p $OUTPUT_PATH

module load toolchain/foss/2022b
module load lib/libaio/0.3.113-GCCcore-12.2.0
module load lang/Python/3.10.8-GCCcore-12.2.0-bare
module load compiler/GCC/10.3.0

# activating venv
source $VENV_PATH/bin/activate

export HF_DATASETS_CACHE=/scratch/hpc-prf-lola/nikit/.cache/huggingface

srun --wait=60 --kill-on-bad-exit=1 python ../tools/hf_ds_to_jsonl.py \
    /scratch/hpc-prf-lola/data/raw_datasets/CulturaX \
    $CUR_LANG \
    train \
    $START_IND \
    $END_IND \
    $OUTPUT_PATH 2>&1

