#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -p hugemem
#SBATCH --exclusive

# Sample usage: sbatch train_gpt_tokenizer.sh

set -eu

export VENV_PATH=~/virt-envs/venv-lola

module load toolchain/foss/2022b
module load lib/libaio/0.3.113-GCCcore-12.2.0
module load lang/Python/3.10.8-GCCcore-12.2.0-bare
module load compiler/GCC/10.3.0

# activating venv
source $VENV_PATH/bin/activate

export HF_DATASETS_CACHE=/scratch/hpc-prf-lola/nikit/.cache/huggingface

srun --wait=60 --kill-on-bad-exit=1 python train_gpt_tokenizer.py