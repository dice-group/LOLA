#!/bin/bash
module load toolchain/foss/2022b
module load lib/libaio/0.3.113-GCCcore-12.2.0
module load lang
module load Anaconda3
module load system/CUDA/12.0.0
module load lib/NCCL/2.16.2-GCCcore-12.2.0-CUDA-12.0.0
module load compiler/GCC/10.3.0

. task.config
export HF_DATASETS_CACHE=/scratch/hpc-prf-lola/nikit/.cache/huggingface
# Activate the virtual environment
CONDA_VENV_DIR=$(realpath ./$TASK_NAME-eval)
source activate ./$TASK_NAME-eval

export LD_LIBRARY_PATH=$CONDA_VENV_DIR/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# export this variable to your environment before running this script: export HF_LOLA_EVAL_AT=<your-access-token-here>
huggingface-cli login --token $HF_LOLA_EVAL_AT

MODELS_DL_LOG_DIR=./model_download_logs
mkdir -p $MODELS_DL_LOG_DIR


models=(
    "google/mt5-large"
    "facebook/mbart-large-50"
    "facebook/m2m100_1.2B"
    "bigscience/bloom-1b1"
    "bigscience/bloom-1b7"
    "dice-research/lola_v1"
    "cis-lmu/glot500-base"
    "ai-forever/mGPT"
    "FacebookAI/xlm-roberta-large"
    "SeaLLMs/SeaLLMs-v3-1.5B-Chat"
    "bigscience/bloom-7b1"
    "SeaLLMs/SeaLLM-7B-v2"
    "SeaLLMs/SeaLLM-7B-v2.5"
    "Unbabel/TowerBase-7B-v0.1"
    "HuggingFaceH4/zephyr-7b-beta"
    "MediaTek-Research/Breeze-7B-Base-v1_0"
    "tiiuae/falcon-7b"
    "facebook/xlm-roberta-xl"
    "google/mt5-xl"
)

for model in "${models[@]}"; do
    model_logfile=$(echo "$model" | sed 's|/|__|g')
    export HF_HUB_ENABLE_HF_TRANSFER=1
    sbatch -t 05:00:00 -N 1 -n 1 --mem=20G -J "Downloading HF model: ${model}" --cpus-per-task=4 --output $MODELS_DL_LOG_DIR/$model_logfile.log  huggingface-cli download "$model"
done
