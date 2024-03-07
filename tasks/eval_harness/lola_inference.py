# This code is originally from https://github.com/bigscience-workshop/Megatron-DeepSpeed
# under the license https://huggingface.co/spaces/bigscience/license

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,os.path.pardir)))

from tqdm import tqdm
import torch.nn.functional as F

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

from megatron.utils import get_ltor_masks_and_position_ids

from megatron.initialize import initialize_megatron
import megatron

from tools.convert_checkpoint.deepspeed_checkpoint import DeepSpeedCheckpoint
from tools.convert_checkpoint.deepspeed_to_megatron import _create_rank_checkpoint


import deepspeed


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

def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids

def fetch_model_inputs(text, tokenizer, device):
    
    args = get_args()

    tokens = tokenizer.tokenizer.encode(text)
    
    inp = torch.tensor( tokens, dtype=torch.long).to(device)
    
    inplen, = inp.shape
    
    # since in _collate we make sure length is descending, the longest is always the first one.
    padding_length = inplen
    padding_length = 2048
    # pad to length
    inp = torch.cat([
        inp,  # [seq]
        torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
    ], dim=0)
    
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        inp,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return (tokens, attention_mask, position_ids)



def override_args(args, override_args, skip_keys, skip_if_specified_keys):
    for k, v in vars(override_args).items():
        if k in skip_keys:
            continue
        if k in skip_if_specified_keys and getattr(args, k) is not None:
            continue
        setattr(args, k, v)


def tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='Model conversion options')
    group.add_argument('--output_path', default=None, type=str, help='Output Megatron checkpoint folder')
    group.add_argument('--for_release', action='store_true', help='Convert for release purpose, reset some (progress) counters.')
    return parser

from megatron.arguments import parse_args


def generate_text(input_text, model, tokenizer, max_length=50, device="cuda"):
    """
    Generates text based on the input text for a DeepSpeed-optimized model.

    Args:
        input_text (str): The input text to base the generation on.
        model (deepspeed.DeepSpeedEngine): The DeepSpeed-optimized language model.
        tokenizer: The tokenizer corresponding to the model.
        max_length (int): The maximum length of the sequence to generate.
        device (str): The device to run the generation on, e.g., "cuda".

    Returns:
        str: The generated text.
    """
    tokens, attention_mask, position_ids = fetch_model_inputs(input_text, tokenizer, device)

    input_tokens = torch.tensor( tokens, dtype=torch.long).to(device)
    # Generate text token by token
    with torch.no_grad():
        for step in range(max_length):

            # Forward pass through the model. Adjust according to your model's forward() signature
            output_logits = model(input_tokens, position_ids=position_ids, attention_mask=attention_mask)
            output_logits = output_logits.unsqueeze(0)  # [1, seq, vocab]
            greedy_tokens = output_logits.argmax(dim=-1)

            # Extract the next token (you can also apply temperature sampling here)
            next_token_id = output_logits.argmax(1).unsqueeze(-1)

            # Append the next token ID to the generated sequence
            tokens = torch.cat([tokens, next_token_id], dim=-1)

            # Optionally, stop generating text if the end-of-sequence token is generated

    # Decode the generated ids to text
    generated_text = tokenizer.decode(tokens)

    return generated_text

def main():
    model, ds_checkpoint = load_ds_checkpoint_and_setup_megatron(extra_args_provider=tasks_args)
    
    model.eval()
    tokenizer = get_tokenizer()

    args = get_args()
    
    input_text = "The quick brown fox"
    generated_text = generate_text(input_text, model, tokenizer)
    print("Generated text:", generated_text)
    
    # ds_engine = deepspeed.init_inference(model=model,
    #                                   mp_size=args.tensor_model_parallel_size,
    #                                   tensor_parallel={"mpu": mpu},
    #                                   dtype=torch.half,
    #                                   replace_with_kernel_inject=True,
    #                                   moe_experts=args.num_experts,
    #                                   moe_type=args.mlp_type)
    
    # model = ds_engine.module
    # output = model('Input String')
    # print(output)

if __name__ == '__main__':
    main()
