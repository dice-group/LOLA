#!/bin/bash

module load lib/NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0
module load ai/PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
module load vis/torchvision/0.13.1-foss-2022a-CUDA-11.7.0
# activating venv
source /scratch/hpc-prf-lola/lib_repo/custom-venvs/lola1/bin/activate


python ../../tools/preprocess_data.py \
    --input data/mc4-sample-9k.jsonl \
    --output-prefix data/meg-t5-mc4-sample-9k \
    --dataset-impl mmap \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model data/spiece.model \
    --append-eod \
    --workers 16
