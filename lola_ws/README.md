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



### Downloading CulturaX

**Note:** The scripts provided below are written for noctua2 cluster and have hardcoded paths in them. Please go through them before reusing. 

```bash
# This command might fail from time to time, rerunning it resumes the download
huggingface-cli download uonlp/CulturaX --repo-type dataset
```

Once the download is finished, create a symlink with "CulturaX" as directory name. pointing to your huggingface cache, e.g:
```bash
ln -s /scratch/hpc-prf-lola/nikit/.cache/huggingface/datasets--uonlp--CulturaX/snapshots/321a983f3fd2a929cc1f8ef6207834bab0bb9e25 /scratch/hpc-prf-lola/data/raw_datasets/CulturaX
```

Then run the following command to generate arrow files for all the languages:
```bash
# Note: This command will spawn 167 jobs on your cluster
bash run_process_culturax.sh
```

### Pre-Processing CulturaX

We collected the CulturaX stats in this file: [culturax-v1-0-0_data_stats.json](./gpt/culturax/culturax-v1-0-0_data_stats.json).

We define the percentage of samples we would like to extract samples for prepocessing per language (default applies to non-mentioned languages): [culturax-custom-data-split.json](./gpt/culturax/culturax-custom-data-split.json).

Afterwards, run the following script to submit preprocessing jobs for all languages (1 slurm job per language):

```bash
python3 preprocess_large_data.py
```

The processed datasets will be available at the mentioned `DATA_PATH` in `preprocess_large_data.sh`.


**Note:** In our experience on lustre file system, the steps below degrade the final throughput.

As per the discussion here: https://github.com/NVIDIA/Megatron-LM/issues/452, merging the data into one big file makes sense for some filesystems.
To merge the files, first copy all the `*_text_document` files with `.bin` and `.idx` extension into a single directory and then use the merge tool:

```bash
# copy files for merge
cp -r <path-to-processed-data>/data-*/meg-culturax-*_text_document* <path-to-collect-files-for-merge>
# merge the dataset
sbatch merge_datasets.sh <path-to-collected-files-for-merge> <path-to-output-dir>  meg-culturax
```


