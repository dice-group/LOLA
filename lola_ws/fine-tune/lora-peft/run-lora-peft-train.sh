#!/bin/bash

set -eu

VENV_PATH=./venv-lola-peft

source $VENV_PATH/bin/activate

## Uncomment the statement below to specify CUDA device
export CUDA_VISIBLE_DEVICES=0
## Comment the statement below if wandb needs to be configured
#export WANDB_MODE=offline
export WANDB_PROJECT="LOLA-FineTuning"

postfix=$(date +"%d%m%y%H%M%S")
# Tested with Nvidia H100
torchrun --nnodes=1 --nproc_per_node=1 --master_port=4550 lora-peft-train.py \
    --model_name_or_path dice-research/lola_v1 \
    --data_path ./alpaca_multilingual.json \
    --fp16 True \
    --output_dir ./lola_alpaca_multilingual_peft__attn_rank32 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --logging_steps 10 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --report_to "wandb" \
    --run_name "lora-peft-alpaca-multi-"$postfix

# python -u -m debugpy --wait-for-client --listen 0.0.0.0:12121 -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=4550 lora-peft-train.py \
#     --model_name_or_path dice-research/lola_v1 \
#     --data_path ./alpaca_multilingual.json \
#     --fp16 True \
#     --output_dir ./lola_alpaca_multilingual_peft \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 8 \
#     --save_strategy "steps" \
#     --save_steps 500 \
#     --save_total_limit 1 \
#     --logging_steps 10 \
#     --learning_rate 5e-5 \
#     --weight_decay 0.01 \
#     --warmup_ratio 0.03 \
#     --report_to "wandb" \
#     --run_name "lora-peft-alpaca-multi-"$postfix