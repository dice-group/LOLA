#!/bin/bash
set -eu

PEFT_VENV_NAME=venv-lola-custom-task
PEFT_VENV_DIR=./$PEFT_VENV_NAME

# Tested with python 3.12.3
python -m venv $PEFT_VENV_DIR
source $PEFT_VENV_DIR/bin/activate

# NOTE: Change as per your setup
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

echo "Setup finished. Python virtual environment can be found at: "$PEFT_VENV_DIR