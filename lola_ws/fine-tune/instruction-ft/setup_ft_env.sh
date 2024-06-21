#!/bin/bash
set -eu

## For Python based installation, uncomment below. We have tested it with python 3.10.8
# python -m venv ./venv-lola-ft
# source venv-lola-ft/bin/activate

## For conda based installation, uncomment below.
conda create --prefix ./venv-lola-ft -y python=3.10
source activate ./venv-lola-ft/


pip3 install --upgrade pip

pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 "numpy<1.24" packaging==24.1 datasets==2.20.0 wheel pybind11==2.12.0
pip install transformers[torch]==4.41.2
pip install deepspeed==0.11.1
## Uncomment if wandb is needed
# pip install wandb
## Uncomment if debugpy is needed
# pip install debugpy