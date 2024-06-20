#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 10G
#SBATCH -J "Setting up fine-tuning environment for lola"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
set -eu

# Loading required modules
module load toolchain/foss/2022b
module load lib/libaio/0.3.113-GCCcore-12.2.0
module load lang
module load Anaconda3
module load system/CUDA/12.0.0
module load lib/NCCL/2.16.2-GCCcore-12.2.0-CUDA-12.0.0
module load compiler/GCC/10.3.0

# Calling setup script
bash setup_ft_env.sh