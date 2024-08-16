# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Define the config mapping between HF and modelopt."""

# the map is a list of tuples, and each tuple has two elements
# first element: a list of possible fields on HF
# second element: a name of the layer config field inside modelopt
HF_CONFIG_MAP = [
    (["n_head", "num_attention_heads", "n_heads", "num_heads"], "num_attention_heads"),
    (
        [
            "num_key_value_heads",
            "num_kv_heads",
            "num_kv",
            "n_head_kv",
            "multi_query_group_num",
            "num_query_groups",
            "kv_n_heads",
        ],
        "num_kv_heads",
    ),
    (["n_positions", "max_position_embeddings", "max_seq_len"], "max_position_embeddings"),
    (["rotary_percentage", "rotary_percent", "rotary_pct", "partial_rotary_factor"], "rotary_pct"),
    (["alibi"], "use_alibi"),
    (["alibi_bias_max"], "alibi_bias_max"),
    (["new_decoder_architecture"], "new_decoder_architecture"),  # Falcon
    (["apply_residual_connection_post_layernorm"], "apply_residual_connection_post_layernorm"),
    (["use_cache"], "use_cache"),
    (["rope_ratio"], "rope_ratio"),  # Chatglm
    (["parallel_attn"], "parallel_attention"),
    (["seq_length"], "seq_length"),  # Qwen
    # For Qwen and phi3small rotary_emb_base
    # CodeLlama using different rotary_base in comparison to LLaMA v1/v2 models
    (["rotary_emb_base", "rope_theta", "rotary_base", "rope_embedding_base"], "rotary_base"),
    (["original_max_position_embeddings"], "original_max_position_embeddings"),  # Phi3
    (["partial_rotary_factor"], "partial_rotary_factor"),  # Phi3
    (["mup_attn_multiplier"], "mup_attn_multiplier"),  # Phi3-small
    (["mup_embedding_multiplier"], "mup_embedding_multiplier"),  # Phi3-small
    (["mup_use_scaling"], "mup_use_scaling"),  # Phi3-small
    (["mup_width_multiplier"], "mup_width_multiplier"),  # Phi3-small
    (["blocksparse_block_size"], "blocksparse_block_size"),  # Phi3-small
    (["blocksparse_homo_head_pattern"], "blocksparse_homo_head_pattern"),  # Phi3-small
    (["blocksparse_num_local_blocks"], "blocksparse_num_local_blocks"),  # Phi3-small
    (["blocksparse_vert_stride"], "blocksparse_vertical_stride"),  # Phi3-small
    (["dense_attention_every_n_layers"], "dense_attention_every_n_layers"),  # Phi3-small
    (["gegelu_limit"], "gegelu_limit"),  # Phi3-small
    (
        ["num_local_experts", "moe_num_experts"],
        "moe_num_experts",
    ),  # Mixture of Experts (Mixtral, DBRX)
    (["num_experts_per_tok", "moe_top_k"], "moe_top_k"),  # Mixture of Experts (Mixtral, DBRX)
    (["model_type"], "qwen_type"),  # qwen
    (["lru_width"], "rnn_hidden_size"),  # Recurrent Gemma
    (["embeddings_scale_by_sqrt_dim"], "emb_scale_by_sqrt_dim"),  # Recurrent Gemma
    (["logits_soft_cap"], "logits_soft_cap"),  # Recurrent Gemma
    (["_block_types"], "layer_types"),  # Recurrent Gemma
    (["final_logit_softcapping"], "final_logit_softcapping"),  # Gemma2
    (["attn_logit_softcapping"], "attn_logit_softcapping"),  # Gemma2
    (["query_pre_attn_scalar"], "query_pre_attn_scalar"),  # Gemma2
    (["clip_qkv"], "clip_qkv"),  # DBRX
    (["use_scaled_rope"], "use_scaled_rope"),  # Evian 2
]
