#!/bin/bash
set -eu

LOLA_FT_ENV_NAME=venv-lola-peft
## For Python based installation, uncomment below. We have tested it with python 3.10.8
python -m venv ./$LOLA_FT_ENV_NAME
source $LOLA_FT_ENV_NAME/bin/activate


pip3 install --upgrade pip

#pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<1.24" packaging==24.1 datasets==2.20.0 wheel pybind11==2.12.0
pip install transformers[torch]==4.41.2
pip install peft
#pip install deepspeed==0.11.1
# To download the models
pip install hf_transfer

## Uncomment below two lines if you plan to use python notebooks
pip install ipykernel
ipython kernel install --user --name=$LOLA_FT_ENV_NAME

## Uncomment if wandb is needed
pip install wandb
## Uncomment if debugpy is needed
# pip install debugpy