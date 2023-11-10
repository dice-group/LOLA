#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -N 32
#SBATCH -n 64
#SBATCH -J "Preparing large data"

set -eu

export VENV_PATH=~/virt-envs/venv-lola
export DATA_PATH=./culturax/data1

module load toolchain/foss/2022b
module load lib/libaio/0.3.113-GCCcore-12.2.0
module load lang/Python/3.10.8-GCCcore-12.2.0-bare
module load compiler/GCC/10.3.0

export LD_LIBRARY_PATH=$VENV_PATH/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# activating venv
source $VENV_PATH/bin/activate

LIB_DIR=/scratch/hpc-prf-lola/nikit/repos/LOLA-Megatron-DeepSpeed

mkdir -p $DATA_PATH

wget -N https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -O $DATA_PATH/gpt2-vocab.json
wget -N https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -O $DATA_PATH/gpt2-merges.txt

export HF_DATASETS_CACHE=/scratch/hpc-prf-lola/nikit/.cache/huggingface

srun python ${LIB_DIR}/tools/preprocess_data_dist.py \
    --input uonlp/CulturaX \
    --split "train[0%:1%]"\
    --scratch /dev/shm \
    --output-prefix $DATA_PATH/meg-culturax-1percent \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file $DATA_PATH/gpt2-merges.txt \
    --vocab-file $DATA_PATH/gpt2-vocab.json \
    --append-eod 2>&1