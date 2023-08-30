"""
The code below is inspired from https://github.com/bigscience-workshop/bigscience/blob/58d99c67f643d27b5765a73a2ee2d1ce0a4b2c6b/experiments/gpt2-utils.md.
We have explained each and every variable that is part of the final parameter count.
Additionally, we have introduced variables to count the parameters in a Mixture of Experts setting.

Example usage 1 (6.7B dense):
python gpt_param_estimate_util.py --vocab_size 50257 --hidden_size 4096 --seq_length 2048 --num_heads 32 --num_layers 32

Example usage 2 (6.7B + 128 MoE)
python3 gpt_param_estimate_util.py --vocab_size 50257 --hidden_size 4096 --seq_length 2048 --num_heads 32 --num_layers 32 --is_moe --num_experts 128
"""

import argparse


def calculate_gpt2_parameter_count(v, h, s, n, k, is_moe, num_experts):
    """
    Calculate the parameter count for the GPT2 implementation.

    :param v: Vocabulary size.
    :param h: Hidden size (dimension).
    :param s: Maximum sequence length.
    :param n: Number of transformer blocks (layers).
    :param k: Number of attention heads.
    :param is_moe: If mixture of experts is enabled.
    :param num_experts: Number of experts.
    :return: total_params (int), total_expert_params (int): Total number of dense parameters and the expert
    parameters in the GPT2 model.
    """

    # Compute Embedding Parameters (Vocab + Position)
    emb_params = (v * h) + (s * h)  # Total parameters in the embedding layer (Vocabulary + Positional embeddings).

    # Compute Parameters per Transformer Block
    head_dim = h // k  # Dimension size of each attention head.
    qkv_params_w = k * (
            3 * (h * head_dim))  # Parameters for Query, Key, and Value linear projections (weight). 3 * (h^2)
    mh_reduce_w = (k * head_dim * h)  # Parameters for reducing multi-head attention output (weight). h^2
    qkv_params_b = k * (3 * head_dim)  # Parameters for Query, Key, and Value linear projections (bias). 3 * h
    mh_reduce_b = h  # Parameters for reducing multi-head attention output (bias). h
    pos_ff_exp_w = h * (4 * h)  # Parameters for position-wise feedforward layer (expansion weight). 4 * h^2
    pos_ff_con_w = (4 * h) * h  # Parameters for position-wise feedforward layer (constriction weight). 4 * h^2
    pos_ff_exp_b = 4 * h  # Parameters for position-wise feedforward layer (expansion bias). 4 * h
    pos_ff_con_b = h  # Parameters for position-wise feedforward layer (constriction bias). h
    layer_norm1 = 2 * h  # Parameters for the first layer normalization. 2 * h
    layer_norm2 = 2 * h  # Parameters for the second layer normalization. 2 * h

    # Magic Formula:
    """
    Total number of parameters in the entire GPT2 model, computed by multiplying the parameters in each block
    (12 * (h^2) + 13 * h) by the number of layers (n) and adding the parameters in the embedding layer ((v * h) + (s * h)), as well as output layer (2*h).
    P.S: Not sure about the 2*h part in the end.
    """
    # The formula below will translate to n * (12 * (h ** 2) + 13 * h) + emb_params + 2 * h
    param_count = (n * (
            qkv_params_w + mh_reduce_w + qkv_params_b + mh_reduce_b + pos_ff_exp_w + pos_ff_con_w + pos_ff_exp_b + pos_ff_con_b + layer_norm1 + layer_norm2)) + emb_params + 2 * h

    tot_expert_params = 0
    if is_moe:
        # hidden layer
        expert_ffn_layer1_w = h * 4 * h
        expert_ffn_layer1_b = 4 * h
        # output layer
        expert_ffn_layer2_w = 4 * h * h
        expert_ffn_layer2_b = h

        # number of params in a single expert would translate to: 8h*h + 5*h
        single_expert_param = expert_ffn_layer1_w + expert_ffn_layer1_b + expert_ffn_layer2_w + expert_ffn_layer2_b
        # total expert params (divided by 2 since the expert layer is only put after every other transformer layer)
        tot_expert_params = (n // 2) * num_experts * single_expert_param

    return param_count, tot_expert_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate GPT-2 Model Parameter Count")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size (v)")
    parser.add_argument("--hidden_size", type=int, required=True, help="Hidden size (h)")
    parser.add_argument("--seq_length", type=int, default=2048, help="Maximum sequence length (s)")
    parser.add_argument("--num_heads", type=int, required=True, help="Number of attention heads (k)")
    parser.add_argument("--num_layers", type=int, required=True, help="Number of transformer blocks (n)")

    parser.add_argument("--is_moe", action=argparse.BooleanOptionalAction, help="Is a Mixture of Experts model")
    parser.add_argument("--num_experts", type=int, default=0, help="Number of experts (MoE based config)")

    args = parser.parse_args()

    total_params, total_expert_params = calculate_gpt2_parameter_count(args.vocab_size, args.hidden_size, args.seq_length, args.num_layers,
                                                  args.num_heads, args.is_moe, args.num_experts)

    print(f"Total estimated parameters in the Dense GPT-2 model: {total_params} ({total_params / 10 ** 9 :.2f}B)")
    if args.is_moe:
        total_sparse_params = total_expert_params + total_params
        print(f"Total Estimated Parameters in the Sparse(MoE) GPT-2 model: {total_sparse_params} ({total_sparse_params / 10 ** 9 :.2f}B)")

