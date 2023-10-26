module load ai/PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
python -m venv --system-site-packages ./lola1
source /scratch/hpc-prf-lola/lib_repo/custom-venvs/lola1/bin/activate
pip install transformers
pip install "numpy<1.24"
pip install packaging datasets nltk deepspeed tensorboard
pip install wheel
pip install pybind11

Install Apex:
https://github.com/NVIDIA/apex#linux

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
