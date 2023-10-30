#!/bin/bash
###### Sample usage ######
# noctua2: bash run-gpt3-moe-pretrain.sh --slurm --nnodes=1 --runtime="50:00:00" --static_gate --model_size="1.3" --mb_size=8
# experiment servers (single server): bash run-gpt3-moe-pretrain.sh --nnodes=1 --gpus_per_node=2 --node_rank=0 --static_gate --model_size="1.3" --mb_size=24
# experiment servers (multi-server):
#   Server1 (master node): bash run-gpt3-moe-pretrain.sh --nnodes=2 --gpus_per_node=2 --node_rank=0 --static_gate --model_size="1.3" --mb_size=24
#   Server2 (worker node): bash run-gpt3-moe-pretrain.sh --nnodes=2 --gpus_per_node=2 --node_rank=1 --master_addr=<server1-hostname> --master_port=6005 --rdzv_id=<generated-id-from-server1> --static_gate --model_size="1.3" --mb_size=24
######
# Collect command line arguments
# Default values for variables
export STATIC_GATE=false
export NNODES=1
export GPUS_PER_NODE=4
# Node rank is only needed when not running on SLURM. On SLURM, this value is extracted automatically from SLURM_PROCID.
export NODE_RANK=0
export DGX_NODE=false
export SLURM=false
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=6005
export RUNTIME="100:00:00"
export RDZV_ID=$RANDOM

# Default (dense) Model size (number of parameters in billions)
## Only values that will work: 0.125, 0.35, 0.76, 1.3, 2.7, 6.7, 13 or 175
export MODEL_SIZE=1.3
# Default batch size per GPU
export MICRO_BATCH_SIZE=24
# Number of tokens to train for
export TRAIN_TOKENS=3000000000 # 3B

## Number of experts. EP_SIZE 1 means dense model without MoE
export EP_SIZE=4

## The NAME_ID variable is used for generating a unique name for the model. It acts like a model name prefix.
## When kept the same as a previously trained model (together with other hyperparams), the training will resume from the last checkpoint automatically.
## Please note that other hyperparameters are also used in generation of a unique name, check in the gpt3-moe-pretrain.sh script to see how "NAME" is formed.
export NAME_ID="gpt-normal-moe"

POSITIONAL=()

# Process command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --static_gate|--static_gate=true)  # Boolean static_gate
            export STATIC_GATE=true
            export NAME_ID="gpt-staticgate-moe"
            shift
            ;;
        --dgx_node|--dgx_node=true)  # Boolean dgx_node
            export DGX_NODE=true
            shift
            ;;
        --slurm|--slurm=true)  # Boolean dgx_node
            export SLURM=true
            shift
            ;;
        --nnodes=*)  # Option with argument
            export NNODES="${1#*=}"
            shift
            ;;
        --gpus_per_node=*)  # Option with argument
            export GPUS_PER_NODE="${1#*=}"
            shift
            ;;
        --node_rank=*)  # Option with argument
            export NODE_RANK="${1#*=}"
            shift
            ;;
        --master_addr=*)  # Option with argument
            export MASTER_ADDR="${1#*=}"
            shift
            ;;
        --master_port=*)  # Option with argument
            export MASTER_PORT="${1#*=}"
            shift
            ;;
        --rdzv_id=*)  # Option with argument
            export RDZV_ID="${1#*=}"
            shift
            ;;
        --train_tokens=*)  # Option with argument
            export TRAIN_TOKENS="${1#*=}"
            shift
            ;;
        --model_size=*)  # Option with argument
            export MODEL_SIZE="${1#*=}"
            shift
            ;;
        --mb_size=*)  # Option with argument
            export MICRO_BATCH_SIZE="${1#*=}"
            shift
            ;;
        --ep_size=*)  # Option with argument
            export EP_SIZE="${1#*=}"
            shift
            ;;
        --runtime=*)  # Option with argument
            export RUNTIME="${1#*=}"
            shift
            ;;
        --)  # End of options, start of positional parameters
            shift
            break
            ;;
        *)  # Positional parameter
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

export OUTPUT_DIR=`pwd`
# output path
export OUTPUT_BASEPATH=$OUTPUT_DIR/$NAME_ID"-output"

export NUM_GPUS=$((NNODES*GPUS_PER_NODE))
# Setting Global batch size in the end, since it relies on micro batch size
export GLOBAL_BATCH_SIZE=$((NNODES*GPUS_PER_NODE*MICRO_BATCH_SIZE))

export NAME_POSTFIX="${NAME_ID}-ms_${MODEL_SIZE}B-bs_${GLOBAL_BATCH_SIZE}-gpus_${NUM_GPUS}"

# Restore positional parameters
set -- "${POSITIONAL[@]}"

mkdir -p "train_logs/"

# If running through SLURM or not
if [[ "$SLURM" == "true" ]]; then
    echo "Submitting job to SLURM."

    export RUN_NAME="noctua2-${NAME_POSTFIX}"
    export WANDB_NAME=$RUN_NAME
    EXTRA_PARAMS=""
    if [[ "$DGX_NODE" == "true" ]]; then
        EXTRA_PARAMS=" --partition=dgx --qos=devel "
    fi
    sbatch --job-name=$RUN_NAME \
     --nodes=$NNODES \
     --ntasks-per-node=1 \
     --gres=gpu:a100:$GPUS_PER_NODE \
     --time=$RUNTIME \
     --output="train_logs/%x-slurm_%j.out" \
      $EXTRA_PARAMS gpt3-moe-pretrain.sh
else
    echo -e "PyTorch rendezvous id: ${RDZV_ID}\nIf this is the master node, then provide the id above to the worker nodes: --rdzv_id=${RDZV_ID}"
    echo "Starting process for node rank: ${NODE_RANK}"
    export RUN_NAME="dice_exp-${NAME_POSTFIX}"
    export WANDB_NAME=$RUN_NAME
    bash gpt3-moe-pretrain.sh
fi
