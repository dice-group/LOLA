# This code is originally from https://github.com/bigscience-workshop/Megatron-DeepSpeed
# under the license https://huggingface.co/spaces/bigscience/license

from functools import reduce
from logging import logMultiprocessing
import os
import sys
from checkpoint_reshaping_and_interoperability import convert_checkpoint_from_megatron_to_transformers
from transformers import GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,os.path.pardir)))

from lm_eval.models.gpt2 import GPT2LM
from lm_eval import evaluator, tasks, utils
from lm_eval.base import CacheHook
from tqdm import tqdm
import torch.nn.functional as F

from lm_eval.tasks import ALL_TASKS
from pretrain_gpt import model_provider
import numpy as np
import time

import torch
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.core.enums import ModelType
from megatron.core import mpu
from megatron.training import setup_model_and_optimizer, get_model
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.p2p_communication import recv_forward, send_forward
import pickle
import json

from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model.distributed import DistributedDataParallel as LocalDDP
from megatron.model.module import Float16Module
from deepspeed.runtime.pipe import schedule
from deepspeed.accelerator import get_accelerator


from megatron.initialize import initialize_megatron
import megatron

from tools.convert_checkpoint.deepspeed_checkpoint import DeepSpeedCheckpoint
from tools.convert_checkpoint.deepspeed_to_megatron import _create_rank_checkpoint


from collections import OrderedDict

from modeling_lola_gpt2 import LOLALMHeadModel

MODEL_KEY = 'model'
ARGS_KEY = 'args'
LANGUGAGE_MODEL_KEY = 'language_model'
EMBEDDING_KEY = 'embedding'
ENCODER_KEY = 'encoder'
WORD_EMBEDDINGS_FOR_HEAD_KEY = 'word_embeddings_for_head'
WORD_EMBEDDINGS_KEY = 'word_embeddings'
FINAL_LAYER_NORM_KEY ='final_layernorm'
CHECKPOINT_VERSION_KEY = 'checkpoint_version'
CHECKPOINT_VERSION_VALUE = 3.0
ITERATION_KEY = 'iteration'

def override_args(args, override_args, skip_keys, skip_if_specified_keys):
    for k, v in vars(override_args).items():
        if k in skip_keys:
            continue
        if k in skip_if_specified_keys and getattr(args, k) is not None:
            continue
        setattr(args, k, v)


# Note(Hesslow):
# The model loading is a bit convoluted.
# We want to parse out the model arguments from the checkpoint and use those to initialize megatron-ds.
#
# However megatron-ds expects its arguments on the command line.
# And at that point we don't know them.
#
# Instead we use Jasons way: we load the arguments form the checkpoint and then override _parse_args to return whatever args we want.
#
# If the checkpoint is old, some new arguments may have been introduced and the code will expect these arguments to exist.
# In order to support this we _first_ parse the arguments normally, and then override them with the arguments from the checkpoint.
# Keeping the default-value of newer arguments.
#
# We then use the megatron deepspeed converter to load the deepspeed checkpoints as if they we're megatron checkpoints.
def load_ds_checkpoint_and_setup_megatron(extra_args_provider):
    # parse the megatorn args. But wait with initalizing megatron.
    # avoid printing the arguments, since they will later be overridden.
    _print_args = megatron.arguments._print_args
    megatron.arguments._print_args = lambda *_args, **kwarg: None
    args = parse_args(extra_args_provider=extra_args_provider)

    ds_checkpoint = DeepSpeedCheckpoint(args.load,
                                        tp_degree=args.tensor_model_parallel_size,
                                        pp_degree=args.pipeline_model_parallel_size,
                                        no_pp=args.no_pipeline_parallel)


    cp_args = ds_checkpoint.get_args()
    # Merge the current args with the checkpoint args.
    skip_keys = ['world_size', 'rank', 'local_rank','device_count', 'micro_batch_size','global_batch_size', 'batch_size', 'tensorboard_dir', 'deepspeed', 'deepspeed_config',
                     'data_parallel_size', 'pipeline_model_parallel_size', 'tensor_model_parallel_size', 'moe_expert_parallel_size', 'moe_token_dropping', 'load', 'rampup_batch_size', 'iteration', 'inference', 'random_ltd']

    skip_if_specified = ['merge_file', 'vocab_file']

    cp_args.tokenizer_type = 'GPT2BPETokenizer'

    override_args(args, cp_args, skip_keys, skip_if_specified)

    # stop megatron from reparsing the arguments.
    megatron.arguments.parse_args = lambda *_args, **kwarg: args
    megatron.global_vars._ensure_var_is_not_initialized = lambda *_args, **kwarg: None
    megatron.global_vars._GLOBAL_ARGS = args

    initialize_megatron(extra_args_provider=extra_args_provider)
    megatron.global_vars._GLOBAL_ARGS = args
    torch.distributed.barrier()

    # Initializing megatron will update eg. tokenizer size. Override again.
    override_args(args, cp_args, skip_keys, skip_if_specified)

    # print final arguments.
    _print_args("eval_harness arguments", args)
    if args.deepspeed:

        # Hack #3:
        # Loading pipelined models in deepspeed with different TP than it was originally trained on fails
        # due to a sanity check, that makes sure that all state_dicts that we merge contains attention layers.
        # This, however, is not true for pipelining when we will merge the state_dict for the embeddings which
        # which does not contain these attention-specific keys.
        #
        # Deepspeed does however manage to load the model if we just turn off this sanity check.
        import deepspeed
        deepspeed.runtime.state_dict_factory.MegatronSDLoader.sanity_check = lambda self, ckpt_file_name: None


        cp_path = args.load
        args.load = None
        model, _, _ = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)
        model = model[0]
        zero_enabled = model._config.zero_enabled
        model._config.zero_enabled = False
        _, _ = model.load_checkpoint(cp_path, tag = '.', load_optimizer_states=False, load_lr_scheduler_states=False, load_module_only=True)
        model._config.zero_enabled = zero_enabled
    else:
        model = get_model(model_provider)[0]
        # Initialize megatron model using the parsed state dict.
        sd = _create_rank_checkpoint(ds_checkpoint, None, mpu.get_tensor_model_parallel_rank(), mpu.get_pipeline_model_parallel_rank(), True)

        model.load_state_dict(sd['model'], strict=True)

    torch.distributed.barrier()
    # LOLA: Attempting to save fp16 model
    #model.save_fp16_model(save_dir="/scratch/hpc-prf-lola/nikit/repos/LOLA-Megatron-DeepSpeed/lola_ws/gpt/sample-output")
    return model, ds_checkpoint




def _convert_ds_transformer_state(sd_list):
    new_sd = OrderedDict()
    for i, sd in enumerate(sd_list):
        for key, value in sd.items():
            new_key = f'layers.{i}.{key}'
            new_sd[new_key] = value

    return new_sd

def _create_checkpoint_paths(base_folder, iteration, tp_degree, pp_degree):
    path_list = []
    iter_folder = f'iter_{iteration:07d}'
    for i in range(0, tp_degree):
        path_list.append([])
        for j in range(0, pp_degree):
            rank_folder = f'mp_rank_{i:02d}' if pp_degree == 1 else f'mp_rank_{i:02d}_{j:03d}'
            ckpt_path = os.path.join(rank_folder, 'model_optim_rng.pt')
            path_list[i].append(os.path.join(base_folder, iter_folder, ckpt_path))

    return path_list


def _create_megatron_dict():
    language_model_dict = {
        EMBEDDING_KEY: {},
        ENCODER_KEY: {}
    }
    megatron_dict = {
        MODEL_KEY: {LANGUGAGE_MODEL_KEY: language_model_dict},
        CHECKPOINT_VERSION_KEY: CHECKPOINT_VERSION_VALUE
    }
    return megatron_dict


def _save_checkpoint(file_path, chkpt_sd):
    dir, _ = os.path.split(file_path)
    os.makedirs(dir, exist_ok=True)
    torch.save(chkpt_sd, file_path)


def _renest_sd(sd):
    new_sd = OrderedDict()
    for key, value in sd.items():
        a, b = key.split('.')
        new_sd[a] = {b: value}
    return new_sd

def override_args(args, override_args, skip_keys, skip_if_specified_keys):
    for k, v in vars(override_args).items():
        if k in skip_keys:
            continue
        if k in skip_if_specified_keys and getattr(args, k) is not None:
            continue
        setattr(args, k, v)


def _create_rank_checkpoint_lola(model, args, iteration, for_release=False):
    meg_encoder_sd = OrderedDict()
    meg_embedding_sd = OrderedDict()
    meg_embedding_for_head_sd = OrderedDict()
    
    # extract lola sd
    lola_encoder_sd = model.module.language_model.encoder.state_dict()
    lola_embedding_sd = model.module.language_model.embedding.state_dict()
    
    # meg_encoder_sd.update(_convert_ds_transformer_state(transformer_sd))
    meg_encoder_sd.update(lola_encoder_sd)
    meg_embedding_sd.update(lola_embedding_sd)

    #embedding_sd = ds_checkpoint.get_embedding_state(tp_index)
    #nested_embedding_sd = _renest_sd(embedding_sd)
    #meg_embedding_sd.update(nested_embedding_sd)

    for key, value in lola_embedding_sd.items():
        if key.startswith(WORD_EMBEDDINGS_KEY):
            fields = key.split('.')
            new_fields = fields[1:]
            new_key = '.'.join(new_fields)
            meg_embedding_for_head_sd[new_key] = value

            #final_norm_sd = ds_checkpoint.get_final_norm_state(tp_index)
            #new_final_norm_sd = {f'{FINAL_LAYER_NORM_KEY}.{key}': value for key, value in final_norm_sd.items()}
            #meg_encoder_sd.update(new_final_norm_sd)

    checkpoint_sd = _create_megatron_dict()

    checkpoint_sd[ITERATION_KEY] = iteration
    
    checkpoint_sd[MODEL_KEY][LANGUGAGE_MODEL_KEY][EMBEDDING_KEY] = meg_embedding_sd
    checkpoint_sd[MODEL_KEY][LANGUGAGE_MODEL_KEY][ENCODER_KEY] = meg_encoder_sd
    checkpoint_sd[MODEL_KEY][WORD_EMBEDDINGS_FOR_HEAD_KEY] = meg_embedding_for_head_sd

    checkpoint_sd[ARGS_KEY] = args
    # Adjust specific fields
    checkpoint_sd[ARGS_KEY].tensor_model_parallel_size = 1
    checkpoint_sd[ARGS_KEY].pipeline_model_parallel_size = 1
    if for_release:
        checkpoint_sd[ARGS_KEY].consumed_train_samples = 0
        checkpoint_sd[ARGS_KEY].consumed_valid_samples = 0

    return checkpoint_sd


def _create_latest_file(base_folder, iteration):
    file_path = os.path.join(base_folder, 'latest_checkpointed_iteration.txt')
    os.makedirs(base_folder, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(str(iteration))



def tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='Model conversion options')
    group.add_argument('--output_path', default=None, type=str, help='Output Megatron checkpoint folder')
    group.add_argument('--for_release', action='store_true', help='Convert for release purpose, reset some (progress) counters.')
    return parser


def generate_hf_model_text(inp_text, max_length, tokenizer, model):
    inputs = tokenizer(inp_text, return_tensors="pt").to(model.device)
    output_sequences = model.generate(input_ids=inputs['input_ids'], max_length=max_length)

    # Decode the generated indices to text
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text

from megatron.arguments import parse_args

def main():
    ### Step 1: Convert from DeepSpeed to Megatron

    model, ds_checkpoint = load_ds_checkpoint_and_setup_megatron(extra_args_provider=tasks_args)

    args = get_args()
    
    output_dir = args.output_path
    for_release = args.for_release
    
    print(f'Converting DeepSpeed checkpoint in {args.load} to Megatron checkpoint in {output_dir}')
    # LOLA: Modifying below initialization for MoE model
    iteration = ds_checkpoint.get_iteration()
    _create_latest_file(args.output_path, iteration)
    # there's only 1 checkpoint path as tp_degree and pp_degree is 1 for LOLA model (MoE)
    checkpoint_paths = _create_checkpoint_paths(output_dir, iteration=iteration, tp_degree=1, pp_degree=1)
    print('LOLA: checkpoint_paths:',checkpoint_paths)
            
    sd = _create_rank_checkpoint_lola(model, args, iteration, for_release)
    # Saving intermediate Megatron model
    _save_checkpoint(checkpoint_paths[0][0], sd)
    
    # Sending the loaded weights to garbage collection
    sd = None
    model = None

    ### Step 2: Convert from Megatron to Huggingface

    conversion_args_dict = {
        'megatron_path': '/data/nikit_ws/LOLA-Megatron-DeepSpeed',
        'load_path': output_dir + '/iter_0296000',
        'save_path': output_dir + '/lola_hf_model',
        'tokenizer_name': 'ai-forever/mGPT',
        'max_shard_size': '10GB',
        'print_checkpoint_structure': True
    }

    conversion_args = types.SimpleNamespace(**conversion_args_dict)

    convert_checkpoint_from_megatron_to_transformers(conversion_args)
    
    print('LOLA: model conversion finished, model saved successfully.')

    ### Step 3: Test the converted model

    # Load the model and tokenizer
    model = LOLALMHeadModel.from_pretrained(conversion_args_dict['save_path']).to("cuda:0")
    # saving model
    # model.save_pretrained("/data/nikit_ws/lola_converted_model/lola_v1_huggingface", from_pt=True)
    #tokenizer = AutoTokenizer.from_pretrained('ai-forever/mGPT')
    tokenizer = AutoTokenizer.from_pretrained(conversion_args_dict['save_path'])
    
    input_text = "The quick brown fox"

    generated_text = generate_hf_model_text(input_text, 100, tokenizer, model)

    print('Input text:', input_text)
    print('Generated text:', generated_text)

if __name__ == '__main__':
    main()
