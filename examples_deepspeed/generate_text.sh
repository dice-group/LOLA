#!/bin/bash
export TORCH_CUDA_ARCH_LIST=8.6+PTX
CHECKPOINT_PATH=/scratch/hpc-prf-lola/nikit/repos/LOLA-Megatron-DeepSpeed/lola_ws/gpt/gpt-normal-16moe-output/checkpoint/noctua2-gpt-normal-16moe_ms-1.3B_bs-768_gpus-96_lr-2.0e-4_minlr-2.0e-5_ep-16_mlc-0.01_cap-1.0_drop-true/
VOCAB_FILE=/scratch/hpc-prf-lola/data/misc/mgpt/mgpt_vocab.json
MERGE_FILE=/scratch/hpc-prf-lola/data/misc/mgpt/mgpt_merges.txt
b=1
mp=1
experts=16
nodes=1
gpus=1


use_tutel=""
#use_tutel="--use-tutel"


ds_inference=""
#ds_inference="--ds-inference"

export CUDA_DEVICE_MAX_CONNECTIONS=1

launch_cmd="deepspeed --num_nodes $nodes --num_gpus $gpus"
L=24
H=2048
A=16
#experts1=${experts[$k]}
program_cmd="tools/generate_samples_gpt.py \
       --tensor-model-parallel-size $mp \
       --num-layers $L \
       --hidden-size $H \
       --num-attention-heads $A \
       --max-position-embeddings 2048 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --num-experts ${experts} \
       --mlp-type standard \
       --micro-batch-size $b \
       --seq-length 2048 \
       --out-seq-length 2048 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --genfile unconditional_samples.json \
       --top_p 0.9 \
       --log-interval 1 \
       --num-samples 0 \
       --load $CHECKPOINT_PATH \
       $use_tutel $ds_inference"

echo $launch_cmd $program_cmd
$launch_cmd $program_cmd
