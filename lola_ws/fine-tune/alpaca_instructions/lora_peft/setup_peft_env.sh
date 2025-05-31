#!/bin/bash
set -eu

LOLA_FT_ENV_NAME=venv-lola-peft
## Tested with python 3.10.8
python -m venv ./$LOLA_FT_ENV_NAME
source $LOLA_FT_ENV_NAME/bin/activate


pip3 install --upgrade pip

pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121 --trusted-host download.pytorch.org

pip install -r requirements.txt

## Uncomment below line if you plan to use python notebooks
ipython kernel install --user --name=$LOLA_FT_ENV_NAME