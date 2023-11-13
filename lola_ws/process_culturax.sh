#!/bin/bash
#SBATCH -t 00:45:00
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G


# Sample usage: sbatch process_culturax.sh

CUR_LANG=$1
NUM_PROCS=$2
SEARCH_DIR=$3

set -eu

export VENV_PATH=~/virt-envs/venv-lola

module load toolchain/foss/2022b
module load lib/libaio/0.3.113-GCCcore-12.2.0
module load lang/Python/3.10.8-GCCcore-12.2.0-bare
module load compiler/GCC/10.3.0

source $VENV_PATH/bin/activate

python process_culturax.py $CUR_LANG $NUM_PROCS $SEARCH_DIR

