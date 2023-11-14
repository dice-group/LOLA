#!/bin/bash
# Deleting previously built files
rm $LIB_DIR/megatron/data/helpers.cpython-310-x86_64-linux-gnu.so
# rm -r $LIB_DIR/megatron/fused_kernels/build/

# activating venv

if [[ "$SLURM" == "true" ]]; then
    #load modules
    module load toolchain/foss/2022b
    module load lib/libaio/0.3.113-GCCcore-12.2.0
    module load lang/Python/3.10.8-GCCcore-12.2.0-bare
    module load system/CUDA/12.0.0
    module load lib/NCCL/2.16.2-GCCcore-12.2.0-CUDA-12.0.0
    # module load compiler/GCCcore/12.3.0
    module load compiler/GCC/10.3.0

    export LD_LIBRARY_PATH=$VENV_PATH/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

    # Currently, the memory copy logic does not work when multiple nodes are involved, since, it creates index-caches at rank 0 at a later stage.
    # echo "Copy data to CPU memory"
    # srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "mkdir -p /dev/shm/lola_data && cp -r $DATA_DIR /dev/shm/lola_data/"
    # wait
    # export DATA_DIR=/dev/shm/lola_data/data
    # echo "Copy done"

    # so processes know who to talk to
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
fi
# activating virtual environment
echo "Activating virtual environment: ${VENV_PATH}/bin/activate"
source $VENV_PATH/bin/activate


###############################################################################
### Main configs
## GPT-3 models use 2K sequence length/context window
SEQ_LEN=2048

### The "GPT-3 XXX" below are configs from GPT-3 paper
### https://arxiv.org/abs/2005.14165, choose based on
### your desired model size or build your own configs


if [[ "$MODEL_SIZE" == "0.125" ]]; then
    ## GPT-3 Small 125M
    NUM_LAYERS=12
    HIDDEN_SIZE=768
    NUM_ATTN_HEADS=12
    #MICRO_BATCH_SIZE=1
    # LR=6.0e-4
    # MIN_LR=6.0e-5
elif [[ "$MODEL_SIZE" == "0.35" ]]; then
    ## GPT-3 Medium 350M
    #MODEL_SIZE=0.35
    NUM_LAYERS=24
    HIDDEN_SIZE=1024
    NUM_ATTN_HEADS=16
    # Use Micro batch size instead of global batch size, the latter is set to $((NNODES*GPUS_PER_NODE*MICRO_BATCH_SIZE))
    #MICRO_BATCH_SIZE=1
    ### GLOBAL_BATCH_SIZE=256
    LR=3.0e-4
    MIN_LR=3.0e-5
elif [[ "$MODEL_SIZE" == "0.76" ]]; then
    ## GPT-3 Large 760M
    #MODEL_SIZE=0.76
    NUM_LAYERS=24
    HIDDEN_SIZE=1536
    NUM_ATTN_HEADS=16
    # Use Micro batch size instead of global batch size, the latter is set to $((NNODES*GPUS_PER_NODE*MICRO_BATCH_SIZE))
    # MICRO_BATCH_SIZE=1
    ### GLOBAL_BATCH_SIZE=256
    LR=2.5e-4
    MIN_LR=2.5e-5
elif [[ "$MODEL_SIZE" == "1.3" ]]; then
    ## GPT-3 XL 1.3B
    #MODEL_SIZE=1.3
    NUM_LAYERS=24
    HIDDEN_SIZE=2048
    NUM_ATTN_HEADS=16
    # Use Micro batch size instead of global batch size, the latter is set to $((NNODES*GPUS_PER_NODE*MICRO_BATCH_SIZE))
    # MICRO_BATCH_SIZE=24
    ### GLOBAL_BATCH_SIZE=512
    LR=2.0e-4
    MIN_LR=2.0e-5
elif [[ "$MODEL_SIZE" == "2.7" ]]; then
    ## GPT-3 2.7B
    #MODEL_SIZE=2.7
    NUM_LAYERS=32
    HIDDEN_SIZE=2560
    NUM_ATTN_HEADS=32
    # Use Micro batch size instead of global batch size, the latter is set to $((NNODES*GPUS_PER_NODE*MICRO_BATCH_SIZE))
    # MICRO_BATCH_SIZE=1
    ### GLOBAL_BATCH_SIZE=512
    LR=1.6e-4
    MIN_LR=1.6e-5
elif [[ "$MODEL_SIZE" == "6.7" ]]; then
    ## GPT-3 6.7B
    #MODEL_SIZE=6.7
    NUM_LAYERS=32
    HIDDEN_SIZE=4096
    NUM_ATTN_HEADS=32
    # Use Micro batch size instead of global batch size, the latter is set to $((NNODES*GPUS_PER_NODE*MICRO_BATCH_SIZE))
    #MICRO_BATCH_SIZE=8
    ### GLOBAL_BATCH_SIZE=1024
    LR=1.2e-4
    MIN_LR=1.2e-5
elif [[ "$MODEL_SIZE" == "13" ]]; then
    ## GPT-3 13B
    #MODEL_SIZE=13
    NUM_LAYERS=40
    HIDDEN_SIZE=5120
    NUM_ATTN_HEADS=40
    # Use Micro batch size instead of global batch size, the latter is set to $((NNODES*GPUS_PER_NODE*MICRO_BATCH_SIZE))
    #MICRO_BATCH_SIZE=1
    ### GLOBAL_BATCH_SIZE=1024
    LR=1.0e-4
    MIN_LR=1.0e-5
elif [[ "$MODEL_SIZE" == "175" ]]; then
    ## GPT-3 175B
    # MODEL_SIZE=175
    NUM_LAYERS=96
    HIDDEN_SIZE=12288
    NUM_ATTN_HEADS=96
    # Use Micro batch size instead of global batch size, the latter is set to $((NNODES*GPUS_PER_NODE*MICRO_BATCH_SIZE))
    # MICRO_BATCH_SIZE=1
    ### GLOBAL_BATCH_SIZE=1536
    LR=0.6e-4
    MIN_LR=0.6e-5
else
    echo "No proper model size provided."
    exit -1
fi

###############################################################################
### Training duration configs
## The main termination condition, original GPT-3 paper trains for 300B tokens
## For MoE model, we found sometimes training a bit more to 330B tokens helps
# TRAIN_TOKENS=300000000000
## TRAIN_ITERS is another termination condition and also affect the number of
## data samples to be indexed. Since we want to reach the TRAIN_TOKENS
## above, and techniques like curriculum learning has less token in some steps,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by TRAIN_ITERS.
TRAIN_ITERS=$(( ${TRAIN_TOKENS} * 3 / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

## Another termination condition in minutes. Set it large enough to avoid
## undesired early termination.
EXIT_DURATION=6000000000
###############################################################################
### LR configs
## LR warmup and decay duration, this token-based config is preferable since
## no need to readjust when the batch size/seqlen is changed.
## Original GPT-3 paper uses 375M warmup tokens and 260B decay tokens.
## For MoE model, we found that setting the decay token to 300B helps.
#WARMUP_TOKENS=375000000
# LR_DECAY_TOKENS=260000000000
#LR_DECAY_TOKENS=300000000000
# LOLA Specific config
# 10% warmup tokens
WARMUP_TOKENS=$(( TRAIN_TOKENS * 10/100 ))
# 80% decay
LR_DECAY_TOKENS=$(( TRAIN_TOKENS * 80/100 ))
###############################################################################
### Parallelism configs
## Micro batch size per GPU
## Make sure that BATCH_SIZE <= GLOBAL_BATCH_SIZE*PP_SIZE*MP_SIZE/NUM_GPUS
# LOLA: We have replaced this with MICRO_BATCH_SIZE
# BATCH_SIZE=4

## Model parallelism, 1 is no MP
## Currently MoE models have divergence issue when MP > 1.
MP_SIZE=1

## Pipeline parallelism
## Currently we don't support PP for MoE. To disable PP, set PP_SIZE
## to 1 and use the "--no-pipeline-parallel" arg.
PP_SIZE=1
###############################################################################
### MoE configs
## Number of experts. EP_SIZE 1 means dense model without MoE
# EP_SIZE=1
# EP_SIZE=4

if [[ $EP_SIZE -gt $NUM_GPUS ]]; then
   EP_PARALLEL_SIZE=$NUM_GPUS
else
   EP_PARALLEL_SIZE=$EP_SIZE
fi

##### Obsolete start #####
# For LOLA, we are keeping EP_PARALLEL_SIZE as 1, to have the full model on each GPU.
## EP_PARALLEL_SIZE=1
##### Obsolete end #####

## Original GPT-3 model always set min LR at 10% of max LR. For MoE model, we
## found that lower LR and min LR (than the base dense model) helps.
## For 1.3B MoE-128 model we used LR=1.2e-4 and MIN_LR=1.0e-6.
## For 350M MoE-128 model we used LR=2.0e-4 and MIN_LR=2.0e-6, but they are not
## heavily tuned.
#LR=2.0e-4
#MIN_LR=2e-06

## Coefficient for MoE loss. We find that 0.01 is a good value at least for
## 1.3B MoE-128 model
MLC=0.01

## Below configs adjust the MoE expert token capacity limit during training and
## eval. To completely disable capacity limit, set MOE_DROP_TOKEN to false.
## Larger capacity factor or disabling capacity limit could improve training
## convergence, but will also reduce training throughput.
MOE_TRAIN_CAP_FACTOR=1.0
MOE_EVAL_CAP_FACTOR=1.0
MOE_MIN_CAP=4
MOE_DROP_TOKEN="true"
# MOE_DROP_TOKEN="false"
###############################################################################
### Curriculum learning (CL) configs
## Enable/disable CL
CL_ENABLED="false"
## Consult the tutorial https://www.deepspeed.ai/tutorials/curriculum-learning/
## for tuning the following configs
CL_START_SEQLEN=80
CL_AVG_SEQLEN=$(( (${CL_START_SEQLEN} + ${SEQ_LEN}) / 2 ))
CL_TOKENS=60
CL_TOKENS=$((${CL_TOKENS} * 1000000000))
CL_STEP=$(( ${CL_TOKENS} / (${GLOBAL_BATCH_SIZE} * ${CL_AVG_SEQLEN}) ))
###############################################################################
### Misc configs
LOG_INTERVAL=5
EVAL_ITERS=200
EVAL_INTERVAL=500
SAVE_INTERVAL=500

## Standard deviation for weight initialization
## We used 0.014 for 350M/1.3B dense/MoE models, and used 0.01 for 6.7B
## dense model. Usually larger model needs lower std.
#INIT_STD=0.014
INIT_STD=0.01

## Activation checkpointing saves GPU memory, but reduces training speed
ACTIVATION_CHECKPOINT="true"
# ACTIVATION_CHECKPOINT="false"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}"

NAME="${RUN_NAME}_lr-${LR}_minlr-${MIN_LR}"

if [[ $EP_SIZE -gt 1 ]]; then
    NAME="${NAME}_ep-${EP_SIZE}_mlc-${MLC}_cap-${MOE_TRAIN_CAP_FACTOR}_drop-${MOE_DROP_TOKEN}"
fi
if [ "${CL_ENABLED}" = "true" ]; then
    NAME="${NAME}_cl-${CL_START_SEQLEN}-${CL_STEP}"
fi

mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
# TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${host}_${current_time}"
# Attaching tensorboard to only the name of the model
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}
## Note that for MoE model with billion-scale base model, the checkpoint can be
## as large as TB-scale which normal NFS cannot handle efficiently.
CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

# The sample data comes from here: https://github.com/nikit91/Megatron-DeepSpeed-BigScience/blob/main/start_fast.md#2-data
#VOCAB_PATH=$DATA_DIR/gpt2-vocab.json
#MERGE_PATH=$DATA_DIR/gpt2-merges.txt
# Public the Pile dataset, can be downloaded at https://mystic.the-eye.eu/public/AI/pile_neox/
#DATA_BLEND=$DATA_DIR/meg-gpt-mc4-1m_text_document
#DATA_BLEND=$DATA_DIR/mc4-gpt-4pt5m_text_document
###############################################################################
data_options=" \
         --vocab-file ${VOCAB_PATH} \
         --merge-file ${MERGE_PATH} \
         --data-path ${DATA_BLEND} \
         --data-impl mmap"

extra_options=""
if [[ "$STATIC_GATE" == "true" ]]; then
    extra_options=" --lola-enable-static-moe-gate"
fi

megatron_options=" \
        --override-opt_param-scheduler \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --tensor-model-parallel-size ${MP_SIZE} \
        --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
        --num-experts ${EP_SIZE} \
        --moe-loss-coeff ${MLC} \
        --moe-train-capacity-factor ${MOE_TRAIN_CAP_FACTOR} \
        --moe-eval-capacity-factor ${MOE_EVAL_CAP_FACTOR} \
        --moe-min-capacity ${MOE_MIN_CAP} \
        --init-method-std ${INIT_STD} \
        --lr-decay-tokens ${LR_DECAY_TOKENS} \
        --lr-warmup-tokens ${WARMUP_TOKENS} \
        --micro-batch-size ${MICRO_BATCH_SIZE} \
        --exit-duration-in-mins ${EXIT_DURATION} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-tokens ${TRAIN_TOKENS} \
        --train-iters ${TRAIN_ITERS} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --split 98,2,0 \
        --log-interval ${LOG_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --save-interval ${SAVE_INTERVAL} \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --hysteresis 2 \
        --num-workers 128 \
        --fp16 \
        --load ${CHECKPOINT_PATH} \
        --save ${CHECKPOINT_PATH} \
        --tensorboard-queue-size 1 \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        ${extra_options}"

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
megatron_options="${megatron_options} \
        --checkpoint-activations"
fi

if [[ $EP_SIZE -gt 1 ]]; then
megatron_options="${megatron_options} \
        --create-moe-param-group"
fi

if [ "${MOE_DROP_TOKEN}" = "false" ]; then
megatron_options="${megatron_options} \
        --disable-moe-token-dropping"
fi

ZERO_STAGE=2

#template_json="ds_config_gpt_TEMPLATE.json"
template_json="${LIB_DIR}/lola_ws/cfg/ds_config_gpt_Zero2_TEMPLATE.json"
config_json="${OUTPUT_BASEPATH}/ds_config_gpt_${NAME}.json"
sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${MICRO_BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/${LOG_INTERVAL}/" \
    | sed "s/ZERO_STAGE/${ZERO_STAGE}/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/CONFIG_FP16_ENABLED/true/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
    | sed "s/CONFIG_CL_ENABLED/${CL_ENABLED}/" \
    | sed "s/CONFIG_CL_MIN/${CL_START_SEQLEN}/" \
    | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
    | sed "s/CONFIG_CL_DURATION/${CL_STEP}/" \
	  > ${config_json}

deepspeed_options=" \
		    --deepspeed \
		    --deepspeed_config ${config_json} \
        --zero-stage ${ZERO_STAGE} \
		    --pipeline-model-parallel-size ${PP_SIZE}"

# Currently MoE is not compatible with pipeline parallel
if [[ $EP_SIZE -gt 1 ]]; then
deepspeed_options="${deepspeed_options} \
        --no-pipeline-parallel"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
        --deepspeed-activation-checkpointing"
fi

# LOLA: some flags from previous setups
# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1
# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE="${OUTPUT_BASEPATH}/torch-elastic-error.json"
export NCCL_ASYNC_ERROR_HANDLING=1


# launcher to python -u -m torch.distributed.run
export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_id=$RDZV_ID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --max_restarts 0 \
    --tee 3 \
    "
export CMD="${LIB_DIR}/pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options}"
# Command for distributed training setup
# run_cmd="deepspeed ${LIB_DIR}/pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options} 2>&1 | tee -a ${OUTPUT_BASEPATH}/log/${NAME}_${host}_${current_time}.log"
echo LAUNCHER: $LAUNCHER
echo CMD: $CMD
export NCCL_DEBUG=TRACE
if [[ "$SLURM" == "true" ]]; then
    ### Workaround for link-flip issue: https://apps.fz-juelich.de/jsc/hps/juwels/known-issues.html#flipping-links
    export NCCL_IB_TIMEOUT=50
    export UCX_RC_TIMEOUT=4s
    export NCCL_IB_RETRY_CNT=10
    ###
    srun --wait=60 --kill-on-bad-exit=1 bash -c "$LAUNCHER --node_rank \$SLURM_PROCID $CMD" 2>&1 | tee -a ${OUTPUT_BASEPATH}/log/${NAME}_${host}_${current_time}.log
else
    $LAUNCHER --node_rank $NODE_RANK $CMD 2>&1 | tee -a ${OUTPUT_BASEPATH}/log/${NAME}_${host}_${current_time}.log
fi
set -x