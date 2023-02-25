#!/bin/bash
#SBATCH -J "Pretrain BERT with the Pile"
#SBATCH -N 2
###SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
###SBATCH -p largemem
#SBATCH --mem-per-cpu 5G
#SBATCH -q express
#SBATCH --gres=gpu:a100:4
#SBATCH -t 60

module load mpi/OpenMPI/4.1.4-GCC-11.3.0
module load system/CUDA/11.7.0
module load lib/NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0

python gen_host_file.py
bash ds_pretrain_bert.sh
