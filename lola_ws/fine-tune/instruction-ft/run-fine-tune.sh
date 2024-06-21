#!/bin/bash

set -eu

VENV_PATH=./venv-lola-ft

## Uncomment below for conda based venv
source activate $VENV_PATH
## Uncomment below for python based venv
#source $VENV_PATH/bin/activate

export LD_LIBRARY_PATH=$VENV_PATH/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

## Uncomment the statement below to specify CUDA device
export CUDA_VISIBLE_DEVICES=0
## Comment the statement below if wandb needs to be configured
#export WANDB_MODE=offline

torchrun --nnodes=1 --nproc_per_node=1 --master_port=4550 train.py \
    --model_name_or_path dice-research/lola_v1 \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir ./lola_alpaca_test \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed default_offload_opt_param.json \
    --tf32 True

# python -u -m debugpy --wait-for-client --listen 0.0.0.0:12121 -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=4545 train.py \
#     --model_name_or_path dice-research/lola_v1 \
#     --data_path ./alpaca_data.json \
#     --bf16 True \
#     --output_dir ./lola_alpaca_test \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 8 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 2000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --deepspeed default_offload_opt_param.json \
#     --tf32 True