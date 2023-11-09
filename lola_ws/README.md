### Virtual-environment setup on Noctua2

Use [setup_slurm.sh](setup_slurm.sh)

### Virtual-environment setup on experiment server / VM
```bash
python -m venv ./venv-lola
source venv-lola/bin/activate
pip3 install torch torchvision torchaudio # torch==2.1.0+cu121 torchvision==0.16.0+cu121
```

#### Dependencies installation

```bash
pip install transformers
pip install "numpy<1.24"
pip install packaging datasets nltk tensorboard deepspeed==0.11.1 # Freezing deepspeed version to make sure our "layer.py" replacement matches
pip install wheel
pip install pybind11
pip install wandb
```

Install Apex:
https://github.com/NVIDIA/apex#linux

```bash
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 4e1ae43
# Note: Comment line 171 in setup.py to avoid legacy installation error.
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

#### To use the the python environment as a notebook kernel (optional)

```bash
pip install ipykernel
python -m ipykernel install --user --name=venv-lola
```

#### Static TopKGate in MoE layer

To introduce the static gating the MoE layer, we have tweaked the logic in the Deepspeed library. At the moment, to use this logic we are using a workaround that involves replacing the original `layer.py` with [gpt/overriden_classes/layer.py)](./gpt/overriden_classes/layer.py). The original file should be located in your virtual environment in a path like this: `venv-lola/lib/<your-python-version>/site-packages/deepspeed/moe/layer.py`. To replace, simply do the following:

```bash
# Backup the original file
mv venv-lola/lib/<your-python-version>/site-packages/deepspeed/moe/layer.py venv-lola/lib/<your-python-version>/site-packages/deepspeed/moe/layer.py_original
# Copy the modified file
cp lola_ws/gpt/overriden_classes/layer.py venv-lola/lib/<your-python-version>/site-packages/deepspeed/moe/
```