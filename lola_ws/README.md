#### Virtual-environment setup on experiment server / VM
```bash
python -m venv ./venv-lola
source venv-lola/bin/activate
pip3 install torch torchvision torchaudio
```


#### Virtual-environment setup on Noctua2
```bash
module load ai/PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
python -m venv --system-site-packages ./lola1
source /scratch/hpc-prf-lola/lib_repo/custom-venvs/lola1/bin/activate
```

#### Dependencies installation

```bash
pip install transformers
pip install "numpy<1.24"
pip install packaging datasets nltk deepspeed tensorboard
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
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
