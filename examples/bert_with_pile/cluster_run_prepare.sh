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

TOTAL_TASKS=$SLURM_NNODES
echo "Total nodes: ${TOTAL_TASKS}"
TOTAL_CHUNKS=30
CHUNK_OFFSET=$((TOTAL_CHUNKS / TOTAL_TASKS))
echo "Chunk offset: ${CHUNK_OFFSET}"
export CHUNK_OFFSET
export TOTAL_TASKS
srun ./run_prepare.sh
