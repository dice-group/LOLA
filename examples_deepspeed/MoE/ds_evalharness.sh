# This is an example zero-shot eval script. Please first read the readme_evalharness.md under the same directory.

CHECKPOINT_PATH=/scratch/hpc-prf-lola/nikit/repos/LOLA-Megatron-DeepSpeed/lola_ws/gpt/gpt-normal-16moe-output/checkpoint/noctua2-gpt-normal-16moe_ms-1.3B_bs-768_gpus-96_lr-2.0e-4_minlr-2.0e-5_ep-16_mlc-0.01_cap-1.0_drop-true/global_step296000
CONFIG_PATH=/scratch/hpc-prf-lola/nikit/repos/LOLA-Megatron-DeepSpeed/lola_ws/gpt/gpt-normal-16moe-output/eval_demo_config.json
RESULT_PATH=/scratch/hpc-prf-lola/nikit/repos/LOLA-Megatron-DeepSpeed/lola_ws/gpt/gpt-normal-16moe-output/sample_eval.log

PP_SIZE=1
TP_SIZE=1
NO_PP="true"
EP_PARALLEL_SIZE=1
# Currently eval harness does not support data parallel
# However, for MoE models it's possible to enable a "fake data parallel"
# in order to load experts on multiple gpus. At the same time, it's not
# real data parallel because we load the same data on all gpus.
# On the other hand, it's better to use less number of gpus than training,
# to reduce communication overhead.
NUM_NODE=1
NUM_GPU_PER_NODE=1

TASKS="webqs"
# WikiText-2, not used in GPT-3 paper but used in GPT-2 paper
# TASKS="wikitext"
# Tasks that appeared in GPT-3 paper (sorted based on the order in paper), plus WikiText-2.
# TASKS="hellaswag,lambada,triviaqa,webqs,winogrande,piqa,arc_challenge,arc_easy,openbookqa,race,boolq,cb,copa,rte,wic,wsc,multirc,record,anli_r1,anli_r2,anli_r3,wikitext"
# All tasks that confirmed to work, there are more tasks on https://github.com/EleutherAI/lm-evaluation-harness that we didn't test.
# TASKS="hellaswag,lambada,triviaqa,webqs,winogrande,piqa,arc_challenge,arc_easy,openbookqa,race,boolq,cb,copa,rte,wic,wsc,multirc,record,anli_r1,anli_r2,anli_r3,wikitext,logiqa,mathqa,mc_taco,mrpc,prost,pubmedqa,qnli,qqp,sciq,sst,wnli"

VOCAB_FILE=/scratch/hpc-prf-lola/data/misc/mgpt/mgpt_vocab.json
MERGE_FILE=/scratch/hpc-prf-lola/data/misc/mgpt/mgpt_merges.txt

# export HF_DATASETS_OFFLINE=1

# Dummy arguments to make megatron happy. No need to configure them.
# The reason we don't need to configure them and many other arguments is
# because the eval framework will read the arguments from checkpoint file.
MEGATRON_REQUIRED_ARGS="\
    --num-layers -1\
    --hidden-size -1\
    --num-attention-heads -1\
    --seq-length -1 \
    --max-position-embeddings -1
"

CMD="../../tasks/eval_harness/evaluate.py \
    --load $CHECKPOINT_PATH\
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE\
    --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
    --vocab-file $VOCAB_FILE\
    --merge-file $MERGE_FILE\
    --micro-batch-size 12\
    --no-load-optim \
    --no-load-rng \
    --inference \
    --disable-moe-token-dropping \
    --tokenizer-type GPT2BPETokenizer \
    --adaptive_seq_len\
    --eval_fp32\
    --task_list $TASKS\
    --results_path $RESULT_PATH \
    --deepspeed \
    --deepspeed_config $CONFIG_PATH \
    $MEGATRON_REQUIRED_ARGS\
    "

if [[ "${NO_PP}" = "true" ]]; then
CMD="${CMD} \
    --no-pipeline-parallel"
fi

export RDZV_ID=$RANDOM
export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=6005

#LAUNCHER="deepspeed --num_nodes $NUM_NODE --num_gpus $NUM_GPU_PER_NODE"
# LAUNCHER="python -u -m torch.distributed.run --nproc_per_node $NUM_GPU_PER_NODE \
#  --nnodes $NUM_NODE --rdzv_id=$RDZV_ID \
#  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#  --max_restarts 0 \
#  --tee 3 "

LAUNCHER="python -u -m debugpy --wait-for-client --listen 0.0.0.0:5678 -m torch.distributed.run "

$LAUNCHER $CMD