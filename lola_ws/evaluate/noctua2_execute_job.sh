#!/bin/bash
#SBATCH -J "LOLA - Tasks Evaluation"
#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 100G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16

# Loading required modules
module load toolchain/foss/2022b
module load lib/libaio/0.3.113-GCCcore-12.2.0
module load lang
module load Anaconda3
module load system/CUDA/12.0.0
module load lib/NCCL/2.16.2-GCCcore-12.2.0-CUDA-12.0.0
module load compiler/GCC/10.3.0

export HF_DATASETS_CACHE=/scratch/hpc-prf-lola/nikit/.cache/huggingface

# Read the input arguments into variables
task=$1
subtask=$2
model_hf_id=$3
language=$4
results_dir=$5

# Change directory to the specified task
cd tasks/$task

# Call the execute script for task
bash execute.sh -m $model_hf_id -s $subtask -l $language -r $results_dir -c
