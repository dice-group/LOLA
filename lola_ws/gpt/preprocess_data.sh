#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J "Preparing sample MC4 data"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=64
#SBATCH --partition=dgx
#SBATCH --qos=devel

module load lib/NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0
module load ai/PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
module load vis/torchvision/0.13.1-foss-2022a-CUDA-11.7.0
# activating venv
source /scratch/hpc-prf-lola/lib_repo/custom-venvs/lola1/bin/activate

LIB_DIR=/scratch/hpc-prf-lola/nikit/repos/Megatron-DeepSpeed-Microsoft

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
