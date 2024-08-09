#!/bin/bash
set -eu


conda create --prefix ./venv-lola-langfamily -y python=3.10
source activate ./venv-lola-langfamily

conda install ipykernel -y
ipython kernel install --user --name=venv-lola-langfamily

pip install SPARQLWrapper