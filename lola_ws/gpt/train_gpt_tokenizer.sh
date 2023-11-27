#!/bin/bash
#SBATCH -t 100:00:00
#SBATCH -N 1
###SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=128
###SBATCH --mem-per-cpu 3G
###SBATCH --reservation=lola_production_run
#SBATCH -p hugemem
#SBATCH --exclusive

# Sample usage: sbatch train_gpt_tokenizer.sh

set -eu

export VENV_PATH=~/virt-envs/venv-lola

module load toolchain/foss/2022b
module load lib/libaio/0.3.113-GCCcore-12.2.0
module load lang/Python/3.10.8-GCCcore-12.2.0-bare
module load system/CUDA/12.0.0
module load compiler/GCC/10.3.0

export LD_LIBRARY_PATH=$VENV_PATH/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# activating venv
source $VENV_PATH/bin/activate

export HF_DATASETS_CACHE=/scratch/hpc-prf-lola/nikit/.cache/huggingface

DATA_DIR=/scratch/hpc-prf-lola/data/culturaX/jsonl
OUTPUT_FILE=/scratch/hpc-prf-lola/data/culturaX/tokenizers/shuffled_culturax.jsonl

export TMPDIR=/scratch/hpc-prf-lola/data/culturaX/tokenizers/temp

mkdir -p $TMPDIR

# sort --parallel=128 -R $DATA_DIR/*.jsonl > $OUTPUT_FILE
srun --wait=60 --kill-on-bad-exit=1 python train_gpt_tokenizer.py