#!/bin/bash
python3 -m venv ./venv-lola-ft
source venv-lola-ft/bin/activate

# downloading LOLA
huggingface-cli download "dice-research/lola_v1"

# downloading multilingual alpaca data
python generate_multilingual_alpaca_json.py
