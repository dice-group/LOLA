# The code is a modified version of script here: https://github.com/tatsu-lab/stanford_alpaca

import copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

import json
import io

from peft import get_peft_model, LoraConfig, TaskType
from typing import Sequence, Dict
from tqdm import tqdm
from task.sample_task import CustomTask

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        )
        for text in tqdm(strings)
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in tqdm(zip(labels, sources_tokenized["input_ids_lens"])):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        global source_target_builder
        super(SupervisedDataset, self).__init__()
        print_rank0("Loading data...")
        list_data_dict = jload(data_path)

        print_rank0("Formatting inputs...")
        
        sources, targets = source_target_builder(list_data_dict, tokenizer)

        print_rank0("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.train_data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator)

# utility functions for torch distributed training
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def print_rank0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

def train():
    global MODEL_HF_ID
    global LOLA_MODEL, LOLA_TOKENIZER
    global OUTPUT_DIR

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    OUTPUT_DIR=training_args.output_dir
    # Check the required training task
    global source_target_builder

    # initializing chosen adapter
    cur_task = CustomTask()
    # Function that builds the source and targets for the model training
    source_target_builder = cur_task.build_source_and_target
    
    MODEL_HF_ID=model_args.model_name_or_path
    
    config = transformers.AutoConfig.from_pretrained(MODEL_HF_ID, trust_remote_code=True)
    
    # NOTE: If your task is simpler, maybe you would like to focus only attention: ['attn.c_attn', 'attn.c_proj', 'wte', 'lm_head'] 
    target_modules=['attn.c_attn', 'attn.c_proj', 'c_proj', 'wte', 'lm_head'] # default for GPT2 based implementations
    
    if MODEL_HF_ID == 'dice-research/lola_v1':
        # disable aux loss addition to the loss
        config.consider_aux_loss = False
    
    LOLA_MODEL = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_HF_ID,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True
    )
    
    LOLA_TOKENIZER = transformers.AutoTokenizer.from_pretrained(
        MODEL_HF_ID,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    # Update tokenizer
    cur_task.before_train(LOLA_MODEL, LOLA_TOKENIZER)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules
    )

    # Apply LoRA to the model
    LOLA_MODEL = get_peft_model(LOLA_MODEL, peft_config)
    LOLA_MODEL.print_trainable_parameters()
    print_rank0(f'Target module names: {LOLA_MODEL.base_model.targeted_module_names}')
    
    data_module = make_supervised_data_module(tokenizer=LOLA_TOKENIZER, data_args=data_args)
    print_rank0('Setting up trainer')
    trainer = Trainer(model=LOLA_MODEL, tokenizer=LOLA_TOKENIZER, args=training_args, **data_module)
    print_rank0('waiting for all devices before starting the training..')
    dist.barrier()
    print_rank0('Starting training..')
    print_rank0('If DeepSpeed is enabled, this may take some time to start..')
    trainer.train()
    #trainer.train(resume_from_checkpoint = True)
    # Wait for other processes to complete
    dist.barrier()
    # Save and test model only for the main rank
    if is_main_process():
        trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
