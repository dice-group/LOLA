# LOLA &mdash; An Open-Source Massively Multilingual Large Language Model

LOLA is a massively multilingual large language model trained on more than 160 languages using a sparse Mixture-of-Experts Transformer architecture. Evaluation results shows competitive performance in natural language generation and understanding tasks. As an open-source model, LOLA promotes reproducibility and serves as a robust foundation for future research.

You can find additional information about the model and its weights at the link provided below:
- Pretrained base model: [https://huggingface.co/dice-research/lola_v1](https://huggingface.co/dice-research/lola_v1) <br>
<!-- - Instructions fine-tuned model: [https://huggingface.co/dice-research/lola_v1_alpaca_instructions](https://huggingface.co/dice-research/lola_v1_alpaca_instructions) -->

**Note**: This repository is a detached fork of [https://github.com/microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed). It contains the training source code for LOLA, which can be mainly found in [lola_ws/](./lola_ws). Some of the implementations from the original source have been modified within this fork for our use-case.

The original README.md can be found here: [archive/README.md](./archive/README.md)


## Citation
If you use this code or data in your research, please cite our work:
```bibtex
@misc{srivastava2024lolaopensourcemassively,
      title={LOLA -- An Open-Source Massively Multilingual Large Language Model}, 
      author={Nikit Srivastava and Denis Kuchelev and Tatiana Moteu Ngoli and Kshitij Shetty and Michael Roeder and Diego Moussallem and Hamada Zahera and Axel-Cyrille Ngonga Ngomo},
      year={2024},
      eprint={2409.11272},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.11272}, 
}
```
