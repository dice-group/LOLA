#!/bin/bash
#SBATCH -J "Deepspeed Megatron: dependencies installation"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH -t 01:00:00

# Sample usage: sbatch setup_slurm.sh ~/virt-envs
set -eu

module load toolchain/foss/2022b
module load lib/libaio/0.3.113-GCCcore-12.2.0
module load lang/Python/3.10.8-GCCcore-12.2.0-bare
module load system/CUDA/12.0.0
module load lib/NCCL/2.16.2-GCCcore-12.2.0-CUDA-12.0.0
# module load compiler/GCCcore/12.3.0 # apex throws compilation error with this compiler
module load compiler/GCC/10.3.0

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

pip3 install torch torchvision torchaudio transformers "numpy<1.24" packaging datasets nltk tensorboard deepspeed==0.11.1 wheel pybind11 wandb mpi4py
# For evaluation
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

cp $LOLA_WS/overriden_classes/layer.py $VENVPATH/venv-lola/lib/python3.10/site-packages/deepspeed/moe/

rm -rf ../../temp_repos
