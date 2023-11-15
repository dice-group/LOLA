#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu 2G

# Sample usage: sbatch merge_datasets.sh /scratch/hpc-prf-lola/data/culturaX/merged/mgpt-tokenized/unmerged /scratch/hpc-prf-lola/data/culturaX/merged/mgpt-tokenized meg-culturax

set -eu

INPUT_PATH=$1
OUTPUT_PATH=$2
DATA_PREFIX=$3

export VENV_PATH=~/virt-envs/venv-lola
# export OUTPUT_PATH=/scratch/hpc-prf-lola/data/culturaX/merged/mgpt-tokenized

mkdir -p $OUTPUT_PATH

module load toolchain/foss/2022b
module load lib/libaio/0.3.113-GCCcore-12.2.0
module load lang/Python/3.10.8-GCCcore-12.2.0-bare
module load compiler/GCC/10.3.0

# activating venv
source $VENV_PATH/bin/activate


srun --wait=60 --kill-on-bad-exit=1 python ../../tools/merge_datasets.py --input $INPUT_PATH  --output-prefix $OUTPUT_PATH/$DATA_PREFIX