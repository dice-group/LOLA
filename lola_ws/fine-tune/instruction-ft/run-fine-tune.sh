#!/bin/bash

source activate ./lola-ft-venv
export LD_LIBRARY_PATH=./lola-ft-venv/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# Uncomment the statement below to specify CUDA device
export CUDA_VISIBLE_DEVICES=1
# Comment the statement below if wandb needs to be enabled
export WANDB_MODE=offline

torchrun --nnodes=1 --nproc_per_node=1 --master_port=4545 train.py \
    --model_name_or_path dice-research/lola_v1 \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir ./lola_alpaca_test \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
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
#     --model_name_or_path mistralai/Mistral-7B-v0.1 \
#     --data_path ./alpaca_data.json \
#     --bf16 True \
#     --output_dir ./output_model \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 2000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --deepspeed default_offload_opt_param.json \
#     --tf32 True