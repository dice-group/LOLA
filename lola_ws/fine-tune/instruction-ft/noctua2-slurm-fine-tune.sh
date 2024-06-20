#!/bin/bash
#SBATCH -t 30:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 250G
#SBATCH -J "fine-tuning lola"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=64

# Sample usage: sbatch noctua2-slurm-fine-tune.sh


# Loading required modules
module load toolchain/foss/2022b
module load lib/libaio/0.3.113-GCCcore-12.2.0
module load lang
module load Anaconda3
module load system/CUDA/12.0.0
module load lib/NCCL/2.16.2-GCCcore-12.2.0-CUDA-12.0.0
module load compiler/GCC/10.3.0

export VENV_PATH=./lola-ft-venv
export LD_LIBRARY_PATH=$VENV_PATH/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# conda activate $VENV_PATH

source activate $VENV_PATH

bash run-fine-tune.sh