#!/bin/bash
#SBATCH -J "Prepare the Pile Dataset"
#SBATCH -N 15
#SBATCH -n 15
#SBATCH --cpus-per-task=4
#SBATCH -p largemem
#SBATCH --mem-per-cpu 20G
###SBATCH -q express
###SBATCH --gres=gpu:a100:4
#SBATCH -t 15:00:00

module load mpi/OpenMPI/4.1.4-GCC-11.3.0
srun ./run_prepare.py
