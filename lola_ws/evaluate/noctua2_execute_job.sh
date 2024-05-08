#!/bin/bash
#SBATCH -J "LOLA - Tasks Evaluation"
#SBATCH -t 150:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 250G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=64

# Loading required modules
module load toolchain/foss/2022b
module load lib/libaio/0.3.113-GCCcore-12.2.0
module load lang
module load Anaconda3
module load system/CUDA/12.0.0
module load lib/NCCL/2.16.2-GCCcore-12.2.0-CUDA-12.0.0
module load compiler/GCC/10.3.0

# TODO: Call the execute script for task.
