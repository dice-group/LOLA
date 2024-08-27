#!/bin/bash

module load toolchain/foss/2022b
module load lib/libaio/0.3.113-GCCcore-12.2.0
module load lang # Uncomment to have conda (also has conda based python)
module load Anaconda3 # Uncomment to have conda (also has conda based python)

# creating a conda environment for the project dependencies to install
conda create --prefix ./lang-group-venv -y python=3.10
source activate ./lang-group-venv

pip install pandas tqdm
# As per README: https://github.com/antonisa/lang2vec/blob/82ab4457ae3a45f552b8d70310ac2a259b44c62a/README.md
git clone https://github.com/antonisa/lang2vec
cd lang2vec
git checkout 82ab4457ae3a45f552b8d70310ac2a259b44c62a
python3 setup.py install