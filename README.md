<p align="center">
  <img src="lola_ws/lola-logo.png"/>
</p>

# LOLA &mdash; An Open-Source Massively Multilingual Large Language Model

LOLA is a massively multilingual large language model trained on more than 160 languages using a sparse Mixture-of-Experts Transformer architecture. Evaluation results shows competitive performance in natural language generation and understanding tasks. As an open-source model, LOLA promotes reproducibility and serves as a robust foundation for future research.

The final model weights, trained using the Deepspeed-Megatron framework, are available at: [https://files.dice-research.org/projects/LOLA/large/global_step296000/](https://files.dice-research.org/projects/LOLA/large/global_step296000/) <br>

Additional information about the model, along with its HuggingFace implementation, can be found at: [https://huggingface.co/dice-research/lola_v1](https://huggingface.co/dice-research/lola_v1) <br>

**<ins>Note</ins>**: This repository is a detached fork of [https://github.com/microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed). It contains the training source code for LOLA, which can be mainly found in [lola_ws/](./lola_ws). Some of the implementations from the original source have been modified within this fork for our use-case.

The original README.md can be found here: [archive/README.md](./archive/README.md)

## What can I do with this repository?

This repository contains various utilities and implementations that can be used within the context of LOLA or adapted for other similar projects. Below is a list of key functionalities provided by this code repository:

### 1. Fine-tune
You can find the scripts for fine-tuning the model in the [lola_ws/fine-tune](lola_ws/fine-tune) directory. We recommend using the [PEFT](https://huggingface.co/docs/transformers/main/en/peft)-based implementation, which trains instructions in the [Alpaca format](https://github.com/tatsu-lab/stanford_alpaca?tab=readme-ov-file#data-release) using [LORAs](https://huggingface.co/docs/diffusers/en/training/lora) on top of our model. These scripts are located here: [lola_ws/fine-tune/lora-peft](lola_ws/fine-tune/lora-peft). The scripts can be easily adapted for other similar (decoder-only) models or datasets.

### 2. Perform Mixture-of-Experts (MoE) Analysis
To conduct your own analysis of the LOLA MoE routing, you can reuse the scripts in [lola_ws/moe_analysis](lola_ws/moe_analysis).  
**Note:** Some scripts are configured for a specific [SLURM](https://slurm.schedmd.com/)-based computing cluster, such as [noctua2](https://pc2.uni-paderborn.de/systems-and-services/noctua-2). Feel free to modify them for your own use case.

### 3. Pretrain
You can pretrain a similar model from scratch or continue training the LOLA model using the script: [lola_ws/gpt/run-gpt3-moe-pretrain.sh](lola_ws/gpt/run-gpt3-moe-pretrain.sh).  
To prepare the [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) dataset for pretraining, refer to this README: [lola_ws/README.md](lola_ws/README.md).

### 4. Reuse Code
If you plan to train your own model using frameworks like [Megatron](https://github.com/NVIDIA/Megatron-LM) or [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed), the scripts in [lola_ws/](lola_ws/) can be especially useful. For preprocessing large datasets, we included a distributed implementation inspired by [Megatron-LM/issues/492](https://github.com/NVIDIA/Megatron-LM/issues/492). This approach significantly improves efficiency on computing clusters with ample CPU resources.


## Citation
If you use this code or data in your research, please cite our work:
```bibtex
@misc{srivastava2024lolaopensourcemassively,
      title={LOLA -- An Open-Source Massively Multilingual Large Language Model}, 
      author={Nikit Srivastava and Denis Kuchelev and Tatiana Moteu Ngoli and Kshitij Shetty and Michael Röder and Hamada Zahera and Diego Moussallem and Axel-Cyrille Ngonga Ngomo},
      year={2024},
      eprint={2409.11272},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.11272}, 
}
```
