#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem 200G
#SBATCH --cpus-per-task=128

module load lang/Python/3.12.3-GCCcore-13.3.0
module load  system/CUDA/12.1.0

bash start_custom_train.sh
