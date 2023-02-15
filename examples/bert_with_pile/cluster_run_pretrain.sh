#!/bin/bash
#SBATCH -J "Pretrain BERT with the Pile"
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --cpus-per-task=2
###SBATCH -p largemem
#SBATCH --mem-per-cpu 2M
#SBATCH -q express
#SBATCH --gres=gpu:a100:2
#SBATCH -t 00:00:05

module load mpi/OpenMPI/4.1.4-GCC-11.3.0
module load system/CUDA/11.7.0

python gen_host_file.py
bash ds_pretrain_bert.sh
