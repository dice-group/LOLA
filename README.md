# LOLA: Large and Open Source Multilingual Language Model

LOLA is a massively multilingual large language model with (sparse) mixture-of-experts (MoE) layers, which is completely open-source. It is trained on over 160 languages and demonstrates strong multilingual performance in comparison to existing models within its size category.

More details about the model alongside it's weights can be found at the link below:
- Pretrained base model: [https://huggingface.co/dice-research/lola_v1](https://huggingface.co/dice-research/lola_v1) <br>
<!-- - Instructions fine-tuned model: [https://huggingface.co/dice-research/lola_v1_alpaca_instructions](https://huggingface.co/dice-research/lola_v1_alpaca_instructions) -->

**Note**: This repository is a detached fork of [https://github.com/microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed). It contains the training source code for LOLA, which can be mainly found in [lola_ws/](./lola_ws). Some of the implementations from the original source have been modified within this fork for our use-case.

The original README.md can be found here: [archive/README.md](./archive/README.md)


## Citation
If you use this code or data in your research, please cite our work:
```bibtex
@misc{srivastava2024lolaopensourcemassively,
      title={LOLA -- An Open-Source Massively Multilingual Large Language Model}, 
      author={Nikit Srivastava and Denis Kuchelev and Tatiana Moteu and Kshitij Shetty and Michael Roeder and Diego Moussallem and Hamada Zahera and Axel-Cyrille Ngonga Ngomo},
      year={2024},
      eprint={2409.11272},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.11272}, 
}
```
