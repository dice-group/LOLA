#!/bin/bash
#SBATCH -t 06:00:00
#SBATCH -n 32
###SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu 25G

# Sample usage: sbatch preprocess_large_data.sh

set -eu

CUR_LANG=$1
START_IND=$2
END_IND=$3

export VENV_PATH=~/virt-envs/venv-lola
#export DATA_PATH=/scratch/hpc-prf-lola/data/culturaX/data-en-full
export DATA_PATH=/scratch/hpc-prf-lola/data/culturaX/data-$CUR_LANG-$START_IND-$END_IND-ind
#export DATA_PATH=/scratch/hpc-prf-lola/data/falcon-small-test-dataset
module load toolchain/foss/2022b
module load lib/libaio/0.3.113-GCCcore-12.2.0
module load lang/Python/3.10.8-GCCcore-12.2.0-bare
module load compiler/GCC/10.3.0

export LD_LIBRARY_PATH=$VENV_PATH/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# activating venv
source $VENV_PATH/bin/activate

LIB_DIR=/scratch/hpc-prf-lola/nikit/repos/LOLA-Megatron-DeepSpeed

# remove if exists
rm -r $DATA_PATH

mkdir -p $DATA_PATH




# wget -N https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -O $DATA_PATH/gpt2-vocab.json
# wget -N https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -O $DATA_PATH/gpt2-merges.txt

ln -s /scratch/hpc-prf-lola/data/misc/gpt2-vocab.json $DATA_PATH/gpt2-vocab.json
ln -s /scratch/hpc-prf-lola/data/misc/gpt2-merges.txt $DATA_PATH/gpt2-merges.txt

export HF_DATASETS_CACHE=/scratch/hpc-prf-lola/nikit/.cache/huggingface

srun --wait=60 --kill-on-bad-exit=1 python ${LIB_DIR}/tools/preprocess_data_dist.py \
    --input "/scratch/hpc-prf-lola/data/raw_datasets/CulturaX" \
    --split "train[$START_IND:$END_IND]" \
    --lang $CUR_LANG \
    --scratch /dev/shm \
    --output-prefix $DATA_PATH/meg-culturax-$CUR_LANG \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file $DATA_PATH/gpt2-merges.txt \
    --vocab-file $DATA_PATH/gpt2-vocab.json \
    --append-eod 2>&1