# This script provides an implementation of GPT2 based mixture-of-experts model.
# Most of its functionality is copied from existing GPT2 implementation on huggingface: https://huggingface.co/docs/transformers/v4.20.1/en/model_doc/gpt2
# MoE layers are inspired by Mixtral: https://huggingface.co/docs/transformers/v4.39.1/en/model_doc/mixtral
# There are however, slight differences in this implementation to adapt it to behave like DeepSpeed Megatron's GPT2 MoE: https://github.com/microsoft/Megatron-DeepSpeed/blob/main/examples_deepspeed/MoE/ds_pretrain_gpt_1.3B_MoE128.sh
# Please note: Most of the the features from DeepSpeed Megatron's GPT MoE are **not** implemented here.

import warnings
from typing import Optional, Tuple, Union

## Uncomment the below three and comment the other import for model conversion
#import sys
# sys.path.append(".")
# from configuration_lola_gpt2 import LOLAConfig

from .configuration_lola_gpt2 import LOLAConfig
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    MoeCausalLMOutputWithPast
)
from transformers.utils import (
    logging
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP, GPT2Block, GPT2PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.modeling_outputs import ModelOutput
import transformers


logger = logging.get_logger(__name__)

expert_analysis_callback = lambda _: None

class LOLADependencyChecker:
    def __init__(self):
        self.expected_versions = {
            "transformers": "4.47.0"
        }
        self.check_dependencies()

    def check_dependencies(self):
        # Check transformers version
        self._check_version("transformers", transformers.__version__)

    def _check_version(self, package_name, installed_version):
        expected_version = self.expected_versions.get(package_name)
        if installed_version != expected_version:
            warnings.warn(
                f"Warning: The installed {package_name} version ({installed_version}) "
                f"differs from the expected version ({expected_version}). "
                "This may lead to unexpected behavior.",
                category=UserWarning
            )

@dataclass
class MoeModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs with potential hidden states and attentions, and includes auxiliary loss.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed):
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed):
            Router logits computed by MoE routers, used to compute the auxiliary loss for Mixture of Experts models.
        aux_loss (`torch.FloatTensor`, *optional*):
            The auxiliary loss computed from the MoE layers, used to encourage balanced expert utilization.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor, ...]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor, ...]] = None
    aux_loss: Optional[torch.FloatTensor] = None

# LOLA
class LOLAModel(GPT2PreTrainedModel):
    
    config_class = LOLAConfig
    
    def __init__(self, config):
        super().__init__(config)
        # Checking dependencies version
        LOLADependencyChecker()
        
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        # To make sure the GPTBlock selects the right attention
        config._attn_implementation='eager'
        self.h = nn.ModuleList([
            GPT2Block(config, layer_idx=i) if i % 2 == 0 else LOLABlock(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    
    def parallelize(self, device_map=None):
        # Check validity of device_map
        warnings.warn(
            "GPT2Model.parallelize is deprecated and will be removed in v5 of Transformers, you should load your"
            " model with device_map='balanced' in the call to from_pretrained. You can also provide your own"
            " device_map but it needs to be a dictionary module_name to device, so for instance {'h.0': 0, 'h.1': 1,"
            " ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    
    def deparallelize(self):
        warnings.warn(
            "Like parallelize, deparallelize is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        aux_losses = []
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            
            if isinstance(block, LOLABlock):
                # Collect auxiliary loss
                aux_loss = outputs[-1]
                aux_losses.append(aux_loss)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Aggregate auxiliary losses
        if aux_losses:
            total_aux_loss = torch.stack(aux_losses).sum()
        else:
            total_aux_loss = None
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            output = (hidden_states, presents, all_hidden_states, all_self_attentions)
            if total_aux_loss is not None:
                output += (total_aux_loss,)
            return tuple(v for v in output if v is not None)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            router_logits=None,  # Include if router_logits are needed
            aux_loss=total_aux_loss,
        )

class LOLABlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        #self.attn = GPT2SdpaAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.moe = LOLAMOE(
            hidden_size,
            inner_dim,
            config,
            config.num_experts,
            k=config.topk,
            # capacity_factor=1.0,
            # min_capacity=4,
            # drop_tokens=False,
            # use_tutel=False,
            # enable_expert_tensor_parallelism=False,
        )

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states, router_logits, aux_loss = self.moe(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs + (aux_loss,)
        else:
            outputs = (hidden_states,) + outputs + (aux_loss,)

        return outputs  # hidden_states, present, (attentions), aux_loss

class LOLAMOE(nn.Module):
    def __init__(self,
                 hidden_size,
                 inner_dim,
                 config,
                 num_experts,
                 k
                 ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_experts = num_experts
        self.top_k = k

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([GPT2MLP(inner_dim, config) for _ in range(self.num_experts)])

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        routing_probabilities = F.softmax(router_logits, dim=1)
        routing_weights, selected_experts = torch.topk(routing_probabilities, self.top_k, dim=-1)
        # Compute Expert Mask
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts)
        expert_mask = expert_mask.sum(dim=1)  # Shape: [batch_size * seq_length, num_experts]

        # Compute Tokens per Expert and Router Probabilities
        token_fraction_per_expert = expert_mask.float().sum(dim=0) / expert_mask.float().sum()
        mean_router_prob_per_expert = routing_probabilities.mean(dim=0)

        # Calculate Auxiliary Loss
        aux_loss = torch.sum(token_fraction_per_expert * mean_router_prob_per_expert) * self.num_experts

        # Proceed with MoE computation as before
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # Process tokens for each expert
        for expert_idx in range(self.num_experts):
            indices = (selected_experts == expert_idx).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue
            current_states = hidden_states[indices]
            current_output = self.experts[expert_idx](current_states)
            current_weights = routing_weights[indices, (selected_experts[indices] == expert_idx).nonzero(as_tuple=True)[1]]
            final_hidden_states.index_add_(0, indices, current_output * current_weights.unsqueeze(-1))

        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        expert_analysis_callback(selected_experts)
        return final_hidden_states, router_logits, aux_loss

class LOLALMHeadModel(GPT2LMHeadModel):
    
    config_class = LOLAConfig

    def __init__(self, config):
        # preventing initiation of GPT2LMHeadModel directly
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = LOLAModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # To add aux loss or not
        self.consider_aux_loss = config.consider_aux_loss
        logger.debug(f'consider_aux_loss is set to {self.consider_aux_loss}')
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,  # Ensure we get a MoeModelOutputWithPast
        )
        hidden_states = transformer_outputs.last_hidden_state
        
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        
        lm_logits = self.lm_head(hidden_states)

        aux_loss = transformer_outputs.aux_loss if hasattr(transformer_outputs, 'aux_loss') else None

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # We can avoid adding aux loss to the total loss if its not needed (e.g. LORA without targeting expert-gating)
            if aux_loss is not None and self.consider_aux_loss:
                loss += self.config.router_aux_loss_coef * aux_loss

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            router_logits=transformer_outputs.router_logits,
        )