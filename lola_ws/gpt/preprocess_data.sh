#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J "Preparing sample MC4 data"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=128
###SBATCH --partition=dgx
###SBATCH --qos=devel

export VENV_PATH=~/virt-envs/venv-lola

module load toolchain/foss/2022b
module load lib/libaio/0.3.113-GCCcore-12.2.0
module load lang/Python/3.10.8-GCCcore-12.2.0-bare
module load system/CUDA/12.0.0
module load lib/NCCL/2.16.2-GCCcore-12.2.0-CUDA-12.0.0
module load compiler/GCCcore/12.3.0

export LD_LIBRARY_PATH=$VENV_PATH/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# activating venv
source $VENV_PATH/bin/activate

LIB_DIR=/scratch/hpc-prf-lola/nikit/repos/LOLA-Megatron-DeepSpeed

set -e
# Expects a data/ directory with already existing jsonl file in it
# To prepare jsonl, use lola_ws/mc4_data_generation.ipynb
# Original ref: https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/e52bdabbde3c6895aceb76c1bced295c2646121f/start_fast.md#2-data

wget -N https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -O data/gpt2-vocab.json
wget -N https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -O data/gpt2-merges.txt

python ${LIB_DIR}/tools/preprocess_data.py \
    --input data/mc4-sample-1m.jsonl \
    --output-prefix data/meg-gpt-mc4-1m \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file data/gpt2-merges.txt \
    --vocab-file data/gpt2-vocab.json \
    --append-eod \
    --workers $SLURM_CPUS_PER_TASK
