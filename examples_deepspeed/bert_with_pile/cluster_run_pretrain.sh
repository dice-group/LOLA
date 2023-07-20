#!/bin/bash
#SBATCH -J "Pretrain BERT with the Pile"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
###SBATCH --partition=dgx
###SBATCH --qos=devel
#SBATCH -t 12:00:00

module load system/CUDA/11.7.0
module load lib/NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0
module load ai/PyTorch/1.12.0-foss-2022a-CUDA-11.7.0

# activating venv
source /scratch/hpc-prf-lola/lib_repo/custom-venvs/lola1/bin/activate

python gen_host_file.py
bash ds_pretrain_bert.sh
