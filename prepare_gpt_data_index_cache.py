# imports
import torch
import math
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, GPTModelPipe
from megatron.model.rotary_pos_embedding import apply_rotary_pos_emb, RotaryEmbedding
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args

from megatron.initialize import initialize_megatron
from megatron.initialize import set_jit_fusion_options

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator
import os
import subprocess


def generate_index_cache():
    """Build train, valid, and test datasets."""
    # parse arguments
    args = get_args()
    
    # Number of train/valid/test samples.
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * \
                 args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [train_samples,
                                  eval_iters * args.global_batch_size,
                                  test_iters * args.global_batch_size]
    print_rank_0(' > datasets target sizes (minimum size):')
    print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
    print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
    print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path)
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds

if __name__ == "__main__":
     # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    # Set pytorch JIT layer fusion options and warmup JIT functions.
    if get_accelerator().device_name() == 'cuda':
        set_jit_fusion_options()
    generate_index_cache()