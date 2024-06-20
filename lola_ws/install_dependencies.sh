#!/bin/bash
# Sample usage: sbatch setup_slurm.sh ~/virt-envs
set -eu

# Check if the first argument is set, if not assign the current directory path
LOLA_WS=$(pwd)
echo "Current lola workspace: ${LOLA_WS}"
VENVPATH="${1:-$(pwd)}"
echo "Path to install virtual environment at: ${VENVPATH}"

mkdir -p $VENVPATH

cd $VENVPATH

python -m venv ./venv-lola

source venv-lola/bin/activate
pip3 install --upgrade pip

pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 transformers "numpy<1.24" packaging datasets nltk tensorboard deepspeed==0.11.1 wheel pybind11 wandb
# For evaluation of LOLA model using the lm-evaluation-harness script in the repository (without converting the original model)
pip install best-download lm-eval==0.3.0 datasets==2.0.0 transformers==4.20.1 huggingface-hub==0.17.0

mkdir -p temp_repos
cd temp_repos

git clone https://github.com/NVIDIA/apex
cd apex
git checkout 4e1ae43
mv setup.py setup.py_old
cp $LOLA_WS/overriden_classes/setup.py .
export LD_LIBRARY_PATH=$VENVPATH/venv-lola/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cp $LOLA_WS/overriden_classes/layer.py $VENVPATH/venv-lola/lib/python*/site-packages/deepspeed/moe/

rm -rf ../../temp_repos
