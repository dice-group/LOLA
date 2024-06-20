#!/bin/bash

conda create --prefix ./lola-ft-venv -y python=3.10

conda activate ./lola-ft-venv/

## Torch dependencies versions:
## torch==2.3.1+cu121
## torchaudio==2.3.1+cu121
## torchvision==0.18.1+cu121
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install transformers[torch]==4.41.2

pip install deepspeed==0.14.3