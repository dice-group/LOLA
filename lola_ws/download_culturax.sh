#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -J "downloading culturax data"


# Sample usage: sbatch download_culturax.sh

set -eu

export VENV_PATH=~/virt-envs/venv-lola

module load toolchain/foss/2022b
module load lib/libaio/0.3.113-GCCcore-12.2.0
module load lang/Python/3.10.8-GCCcore-12.2.0-bare
module load compiler/GCC/10.3.0

source $VENV_PATH/bin/activate

python download_culturax.py

