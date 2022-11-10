#!/bin/bash
#SBATCH -J "Prepare the Pile Dataset"
#SBATCH -N 10
#SBATCH -n 10
#SBATCH --cpus-per-task=1
#SBATCH -p largemem
#SBATCH --mem-per-cpu 100G
#SBATCH -t 100:00:00

module load mpi/OpenMPI/4.1.4-GCC-11.3.0
srun ./run_prepare.py
