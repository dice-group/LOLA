## Steps to Train Alpaca-Based Instruction LoRAs on the LOLA Model

### 1. Install Dependencies
Modify the [setup script](setup_peft_env.sh) as needed, and then run:
```bash
bash setup_peft_env.sh
```
### 2. Set Up Weights & Biases (Wandb) (Optional)
If you wish to use Wandb for experiment tracking, set it up as follows:
```bash
VENV_PATH=./venv-lola-peft
source $VENV_PATH/bin/activate
wandb login
```

### 3. Download Multilingual Alpaca Data
To obtain multilingual Alpaca data, refer to the following script: [../instruction-ft/generate_multilingual_alpaca_json.py](../instruction-ft/generate_multilingual_alpaca_json.py)

### 4. Run the Training Script
Modify the training arguments to suit your setup, and then run the script:
```bash
bash run-lora-peft-train.sh
```

### 5. Inference
To test your model's output in inference mode, you can use the following python notebook: [lora-inference.ipynb](lora-inference.ipynb).
