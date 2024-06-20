#!/bin/bash
#SBATCH -J "Deepspeed Megatron: dependencies installation"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH -t 01:00:00

# Sample usage: sbatch setup_slurm.sh ~/virt-envs
set -eu

module load toolchain/foss/2022b
module load lib/libaio/0.3.113-GCCcore-12.2.0
module load lang/Python/3.10.8-GCCcore-12.2.0-bare
module load system/CUDA/12.0.0
module load lib/NCCL/2.16.2-GCCcore-12.2.0-CUDA-12.0.0
# module load compiler/GCCcore/12.3.0 # apex throws compilation error with this compiler
module load compiler/GCC/10.3.0


./install_dependencies.sh "$@"