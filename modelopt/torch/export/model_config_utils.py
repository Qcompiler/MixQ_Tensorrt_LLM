# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Common utils for the ModelConfig."""

import dataclasses
import math
from typing import Dict, Union, get_args, get_origin

import numpy as np
import torch

from .model_config import (
    KV_CACHE_FP8,
    QUANTIZATION_FP8,
    QUANTIZATION_INT4_AWQ,
    QUANTIZATION_INT8_SQ,
    QUANTIZATION_W4A8_AWQ,
    QUANTIZATION_INT8_MIX,
    DecoderLayerConfig,
    LayernormConfig,
    LinearConfig,
    MLPConfig,
    ModelConfig,
    MOEConfig,
    QKVConfig,
)

# numpy doesn't know bfloat16, define abstract binary type instead
np_bfloat16 = np.dtype("V2", metadata={"dtype": "bfloat16"})


def _numpy_to_torch(x):
    """Convert numpy array to torch tensor."""
    if isinstance(x, torch.Tensor):
        return x

    if x.dtype != np_bfloat16:
        return torch.tensor(x)
    return torch.tensor(x.view(np.int16)).view(torch.bfloat16)


def model_config_to_dict(model_config: ModelConfig) -> dict:
    """Converts the instance to a python dict."""
    assert model_config is not None, "model_config is None"
    return dataclasses.asdict(model_config)


def split_config_and_weights(config, weights: Dict[str, torch.tensor], prefix: str = "transformer"):
    """Util function to split the weights or any torch.Tensor in nested config to weights.

    A weight id starts with transformers or lm_head will also be generated to link the original key to the weights dict.
    The weights in the weights dict are contiguous.
    """
    if isinstance(config, dict):
        for k, v in config.items():
            if k == "lm_head" and "medusa_heads" not in prefix:
                # lm_head is not part of the transformer.
                array_key = k
            elif k == "experts":
                # Omit the 'experts' key that is not in the model name
                array_key = prefix
            elif k == "medusa_heads":
                # medusa_heads is not part of the transformer
                array_key = k
            else:
                array_key = f"{prefix}.{k}"
            if isinstance(v, torch.Tensor):
                weights[array_key] = v
                config[k] = f"{array_key}"
            else:
                split_config_and_weights(v, weights, array_key)
    elif isinstance(config, list):
        for i, v in enumerate(config):
            array_key = f"{prefix}.{i}"
            if isinstance(v, torch.Tensor):
                weights[array_key] = v
                config[i] = f"{array_key}"
            else:
                split_config_and_weights(v, weights, array_key)


def _unified_weights_key(k: str) -> str:
    """Try to unify the weights dict key between old npz and the new safetensors format."""
    prefixes = ["transformer.", "_np:"]
    for prefix in prefixes:
        if k.startswith(prefix):
            k = k[len(prefix) :]

    k = k.replace("final_layernorm", "ln_f")

    return k.replace(":", ".")


def _restore_model_config(model_config, weights: Dict[str, Union[np.ndarray, torch.Tensor]]):
    def _is_tensor_key(k):
        return isinstance(k, str) and _unified_weights_key(k) in weights

    if isinstance(model_config, dict):
        for k, v in model_config.items():
            if _is_tensor_key(v):
                model_config[k] = _numpy_to_torch(weights[_unified_weights_key(v)])
            else:
                _restore_model_config(v, weights)
    if isinstance(model_config, list):
        for i, v in enumerate(model_config):
            if _is_tensor_key(v):
                model_config[i] = _numpy_to_torch(weights[_unified_weights_key(v)])
            else:
                _restore_model_config(v, weights)


def restore_model_config(model_config, weights: Dict[str, Union[np.ndarray, torch.Tensor]]):
    """Recursively restores the model_config from json and loads np.ndarray or torch.Tensor weights from weights."""
    unified_key_weights = {}
    for k, v in weights.items():
        unified_key_weights[_unified_weights_key(k)] = v

    _restore_model_config(model_config, unified_key_weights)


def _from_dict(class_type, data):
    """Helper function to load the data as a class_type. class_type must be a dataclass."""
    if data is None:
        return None

    if get_origin(class_type) == Union:
        # Handle QKV
        if all([key in data for key in ["q", "k", "v"]]):
            # splitted qkv case
            class_type = QKVConfig
        elif all([key in data for key in ["router", "experts"]]):
            # moe
            class_type = MOEConfig
        elif all([key in data for key in ["fc", "gate", "proj"]]):
            # mlp
            class_type = MLPConfig
        else:
            # merged qkv case
            assert "linear_type" in data, f"{data} is not a valid LinearConfig"
            class_type = LinearConfig

    if dataclasses.is_dataclass(class_type):
        fieldtypes = {f.name: f.type for f in dataclasses.fields(class_type)}
        fields_map = {}
        for k, v in data.items():
            if k in fieldtypes:
                # We only handle keys available in the fields.
                # Deprecated fields in the checkpoint will be ignored.
                fields_map[k] = _from_dict(fieldtypes[k], v)
        return class_type(**fields_map)
    elif get_origin(class_type) == list and dataclasses.is_dataclass(get_args(class_type)[0]):
        list_value = []
        for child in data:
            child_class_type = get_args(class_type)[0]
            list_value.append(_from_dict(child_class_type, child))
        return list_value
    else:
        return data


def model_config_from_dict(d: dict) -> ModelConfig:
    """Load a dict to a `ModelConfig` instance."""
    config_type = ModelConfig

    config_type_map = {}
    for t in [ModelConfig, DecoderLayerConfig, LayernormConfig, LinearConfig]:
        config_type_map[t.__name__] = t

    if "__name__" in d:
        config_name = d.pop("__name__")
        try:
            config_type = config_type_map[config_name]
        except Exception as e:
            raise NotImplementedError(f"{config_name} not supported") from e

    return _from_dict(config_type, d)


def pad_weights(weights, tp_size):
    """Returns the padded weights to tp_size."""
    assert len(weights.shape) > 1

    def _pad_size(original_size, tp_size):
        return int(math.ceil(original_size / tp_size) * tp_size)

    original_size = weights.shape[0]
    padded_size = _pad_size(original_size, tp_size)

    if original_size != padded_size:
        pad_width = padded_size - original_size
        return torch.nn.functional.pad(weights, (0, 0, 0, pad_width), "constant", value=0)
    return weights


def merge_qkv(model_config):
    """Merges the qkv fields in model_config from QKVConfig to a single LinearConfig."""
    for decoder_config in model_config.layers:
        for attention_key in ["attention", "self_attention", "cross_attention"]:
            attention = getattr(decoder_config, attention_key, None)
            if attention and isinstance(attention.qkv, QKVConfig):
                splitted_qkv = attention.qkv
                attention.qkv = LinearConfig()
                attention.qkv.weight = splitted_qkv.weight
                attention.qkv.bias = splitted_qkv.bias
                attention.qkv.activation_scaling_factor = splitted_qkv.activation_scaling_factor
                attention.qkv.weights_scaling_factor = splitted_qkv.weights_scaling_factor
                attention.qkv.weights_scaling_factor_2 = splitted_qkv.weights_scaling_factor_2
                attention.qkv.prequant_scaling_factor = splitted_qkv.prequant_scaling_factor
                attention.qkv.awq_block_size = splitted_qkv.awq_block_size


def merge_fc1_gate(model_config):
    """Merges the qkv fields in model_config from QKVConfig to a single LinearConfig."""
    for decoder_config in model_config.layers:
        if isinstance(decoder_config.mlp, MLPConfig):
            if decoder_config.mlp.merged_fc1_gate:
                fc_weight = decoder_config.mlp.fc.weight
                gate_weight = decoder_config.mlp.gate.weight

                fc_bias = decoder_config.mlp.fc.bias
                gate_bias = decoder_config.mlp.gate.bias

                merged_weight = torch.cat([fc_weight, gate_weight], dim=0)
                decoder_config.mlp.fc.weight = merged_weight

                assert (fc_bias is not None and gate_bias is not None) or (
                    fc_bias is None and gate_bias is None
                )

                if fc_bias is not None:
                    decoder_config.mlp.fc.bias = torch.cat([fc_bias, gate_bias], dim=0)

                assert (fc_bias is not None and gate_bias is not None) or (
                    fc_bias is None and gate_bias is None
                )

                assert decoder_config.mlp.fc.linear_type == decoder_config.mlp.gate.linear_type

                assert (
                    decoder_config.mlp.fc.weights_scaling_factor_2 is None
                    and decoder_config.mlp.gate.weights_scaling_factor_2 is None
                ) or torch.equal(
                    decoder_config.mlp.fc.weights_scaling_factor_2,
                    decoder_config.mlp.gate.weights_scaling_factor_2,
                )

                assert (
                    decoder_config.mlp.fc.activation_scaling_factor is None
                    and decoder_config.mlp.gate.activation_scaling_factor is None
                ) or torch.equal(
                    decoder_config.mlp.fc.activation_scaling_factor,
                    decoder_config.mlp.gate.activation_scaling_factor,
                )

                assert (
                    decoder_config.mlp.fc.prequant_scaling_factor is None
                    and decoder_config.mlp.gate.prequant_scaling_factor is None
                ) or torch.equal(
                    decoder_config.mlp.fc.prequant_scaling_factor,
                    decoder_config.mlp.gate.prequant_scaling_factor,
                )

                assert (
                    decoder_config.mlp.fc.awq_block_size == decoder_config.mlp.gate.awq_block_size
                )

                weight_scaling_factor = decoder_config.mlp.fc.weights_scaling_factor
                if weight_scaling_factor is not None:
                    if weight_scaling_factor.numel() != 1:
                        assert (
                            decoder_config.mlp.fc.weights_scaling_factor.shape
                            == decoder_config.mlp.gate.weights_scaling_factor.shape
                        )
                        merged_weight_scaling_factors = torch.cat(
                            [
                                decoder_config.mlp.fc.weights_scaling_factor,
                                decoder_config.mlp.gate.weights_scaling_factor,
                            ],
                            dim=0,
                        )
                        decoder_config.mlp.fc.weights_scaling_factor = merged_weight_scaling_factors
                    else:
                        assert torch.equal(
                            decoder_config.mlp.fc.weights_scaling_factor,
                            decoder_config.mlp.gate.weights_scaling_factor,
                        )
                decoder_config.mlp.gate = None


def to_quantized_weight(
    weight: torch.Tensor, weights_scaling_factor: torch.Tensor, quantization: str
):
    """Converts the weight to the quantized (packed) format."""
    # Convert the tensor to CPU to avoid potential GPU OOM.
    weight = weight.cpu()
    weights_scaling_factor = weights_scaling_factor.cpu()

    assert quantization == QUANTIZATION_INT8_MIX
    if quantization == QUANTIZATION_INT8_MIX:
        return (weight / weights_scaling_factor[:, None]).round().clamp(-128, 127).to(torch.int8)
    if quantization == QUANTIZATION_FP8:
        # safe tensors does not support fp8 yet. So we pack the tensors as int8
        if weight.dim() == 3:
            # for MOE stacked weights
            return (
                (weight / weights_scaling_factor.unsqueeze(-1))
                .to(torch.float8_e4m3fn)
                .view(torch.int8)
            )
        return (weight / weights_scaling_factor).to(torch.float8_e4m3fn).view(torch.int8)

    if quantization == QUANTIZATION_INT8_SQ:
        return (weight / weights_scaling_factor[:, None]).round().clamp(-128, 127).to(torch.int8)

    if quantization in [QUANTIZATION_INT4_AWQ, QUANTIZATION_W4A8_AWQ]:
        out_dim = weight.shape[-2]
        assert (
            out_dim % 2 == 0
        ), f"Cannot pack weight. Out dimension {out_dim} is not an even number."
        in_dim = weight.shape[-1]
        block_size = weight.shape[-1] // weights_scaling_factor.shape[-1]
        int8_tensor = (
            (weight / weights_scaling_factor[..., :, torch.arange(in_dim) // block_size])
            .round()
            .clamp(-8, 7)
            .to(torch.int8)
        )

        if int8_tensor.dim() == 3:
            # Case of MoE, where weights are stacked
            transpose = int8_tensor.permute(0, 2, 1)  # (experts, in_dim, out_dim)
            int8_tensor = transpose.reshape(
                -1,
                in_dim,
                out_dim // 2,
                2,
            )
            int4x2_tensor = (int8_tensor[..., 0] & 0x0F) | (int8_tensor[..., 1] << 4)
            # The shape of the returned weight is (experts, out_dim // 2, in_dim)
            return int4x2_tensor.permute(0, 2, 1).contiguous()

        int8_tensor = int8_tensor.T.reshape(in_dim, out_dim // 2, 2)  # (in_dim, out_dim)
        int4x2_tensor = (int8_tensor[..., 0] & 0x0F) | (int8_tensor[..., 1] << 4)
        # The shape of the returned weight is (out_dim // 2, in_dim)
        return int4x2_tensor.T.contiguous()

    raise NotImplementedError(f"quantization format {quantization} not supported")


def from_quantized_weight(
    weight: torch.Tensor, weights_scaling_factor: torch.Tensor, quantization: str, torch_dtype
):
    """Converts the quantized weight to the target torch_dtype format."""
    if weight.element_size() >= 2 or weights_scaling_factor is None or not quantization:
        # No need to unquantize the weight.
        return weight.to(torch_dtype)

    if quantization == QUANTIZATION_FP8:
        # safe tensors does not support fp8 yet. So we pack the tensors as int8
        return weight.view(torch.float8_e4m3fn).to(torch_dtype) * weights_scaling_factor.to(
            torch_dtype
        )

    if quantization == QUANTIZATION_INT8_SQ:
        return weight.to(torch_dtype) * weights_scaling_factor[:, None].to(torch_dtype)

    raise NotImplementedError(f"quantization format {quantization} not supported")


def pack_linear_weights(model_config: ModelConfig):
    """Packs the quantized linear weights in the model_config to the quantized format."""
    if not model_config.quantization:
        return
    print("pack_linear_weights")
     
    #print(model_config)


    #relative_path = "act_scales/%s.pt"%(model_config._name_or_path.split("/")[-1])

    #relative_path = "act_scales/Llama-2-7b.pt"
    #relative_path = "act_scales/qwen2-7b-instruct.pt"
    relative_path = "act_scales/Qwen2-72B.pt"
    print("-----load----relative_path-",relative_path)
    act_scales = torch.load(relative_path)
    # from safetensors import safe_open
    # awq_model = torch.load("/code/tensorrt_llm/manual_plugin/checkpoint/Llama-2-7b-w4-g128-v2.pt",
    #                       map_location=torch.device('cpu'))
        
    names = [  "self_attn.q_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                ]
    layer_id = -1
    #print(len(model_config.layers))

    torch.cuda.empty_cache()
    layer_id = -1
    for decoder_config in model_config.layers:

        linear_layers = [
                        decoder_config.attention.qkv ,
                        # decoder_config.attention.dense,
                        # decoder_config.mlp.fc,
                        decoder_config.mlp.gate,
                        decoder_config.mlp.proj,
                    ]  
        layer_id += 1
        print("layer id is ", layer_id)
        for i, linear_layer in enumerate(linear_layers): 

            if isinstance(linear_layer, LinearConfig):
                
                
                name = names[i]
                
                layer_scales = act_scales['model.layers.{}.{}'.format(layer_id, name)]

                print("layer_scales")
                 
                linear_layer.weights_scaling_factor =   (torch.max(torch.abs(linear_layer.weight), dim=1)[0].unsqueeze(1) / (
                        127)).to(torch.float16).reshape((linear_layer.weight.shape[0],))

                if linear_layer.weights_scaling_factor is not None:
                    import mixlib
                    from EETQ import quant_weights, preprocess_weights, w8_a16_gemm

                    # 一个int 放在2个half 中
                    int8_weight_cpu = torch.t(linear_layer.weight.data).contiguous().cpu()
                    int8_weight, scales = quant_weights(int8_weight_cpu, torch.int8, False)
                    
                    linear_layer.qweight = mixlib.int8_matrix_to_half(int8_weight.cuda()).cpu()
                    linear_layer.scales = scales
                    print("-----qweight-----")
                    print(linear_layer.qweight.shape)
                    print(linear_layer.scales.shape)

                    fp_features = 128

                    linear_layer.fp_ind =  torch.sort(layer_scales)[1][-fp_features:] 

                    #print([layer_scales[linear_layer.fp_ind]])
                 
                    linear_layer.fp_weight = linear_layer.weight[:, linear_layer.fp_ind]
                    linear_layer.weight[:, linear_layer.fp_ind] *= 0 # setting to zeros
                    # 转成short
                    import mixlib
                    # 一个int 放在2个half 中
                    linear_layer.fp_ind = mixlib.int_to_half(linear_layer.
                    fp_ind.to(torch.int32).cuda()).cpu()
                    
                    linear_layer.weight = to_quantized_weight(
                        linear_layer.weight.cuda(),
                        linear_layer.weights_scaling_factor.cuda(),
                        model_config.quantization,
                    )
                    print("conver to half")
                    linear_layer.weight = mixlib.int8_matrix_to_half(linear_layer.weight.cuda()).cpu()
                    # print("q weight is ")
                    # print(linear_layer.weight)
                    # print("recorver weight is ")
                    # tmp = linear_layer.weight.to(torch.float16) * linear_layer.weights_scaling_factor[:, None].cpu()
                    # print(tmp[0,0:10])
                    # exit

# def pack_linear_weights(model_config: ModelConfig):
#     """Packs the quantized linear weights in the model_config to the quantized format."""

#     def _linear_layer_to_quantized_weight(linear_layers):
#         for linear_layer in linear_layers:
#             if isinstance(linear_layer, LinearConfig):
#                 if linear_layer.weights_scaling_factor is not None:
#                     linear_layer.weight = to_quantized_weight(
#                         linear_layer.weight,
#                         linear_layer.weights_scaling_factor,
#                         model_config.quantization,
#                     )

#     if not model_config.quantization:
#         return

#     attention_key_list = ["attention", "self_attention", "cross_attention"]
#     for decoder_config in model_config.layers:
#         linear_layers = []
#         if any([hasattr(decoder_config, attention_key) for attention_key in attention_key_list]):
#             for attention_key in attention_key_list:
#                 attention = getattr(decoder_config, attention_key, None)
#                 if attention:
#                     linear_layers = [
#                         attention.qkv,
#                         attention.dense,
#                     ]
#         if decoder_config.recurrent:
#             linear_layers = [
#                 decoder_config.recurrent.linear_y,
#                 decoder_config.recurrent.linear_x,
#                 decoder_config.recurrent.linear_out,
#             ]

#         if isinstance(decoder_config.mlp, MOEConfig):
#             if model_config.quantization not in [QUANTIZATION_FP8, QUANTIZATION_INT4_AWQ]:
#                 raise NotImplementedError(
#                     f"MOE quantization for {model_config.quantization} is not supported yet."
#                 )
#             else:
#                 linear_layers.append(decoder_config.mlp.experts.fc)
#                 linear_layers.append(decoder_config.mlp.experts.proj)
#         else:
#             linear_layers.append(decoder_config.mlp.fc)
#             linear_layers.append(decoder_config.mlp.proj)
#             linear_layers.append(decoder_config.mlp.gate)

#         _linear_layer_to_quantized_weight(linear_layers)

#     if model_config.medusa_heads is not None:
#         linear_layers = []

#         for head in model_config.medusa_heads:
#             linear_layers.append(head.lm_head)
#             for layer in head.medusa_layers:
#                 linear_layers.append(layer.linear)

#         _linear_layer_to_quantized_weight(linear_layers)


def naive_quantization(config: ModelConfig):
    """Generates a constant scaling factor (1) with target quantization.

    This is for debugging and performance measurement only.
    """
    if isinstance(config.layers[0], DecoderLayerConfig):
        assert (
            config.layers[0].decoder_type != "mixtral"
        ), "Mixtral native quantization is not supported."

    config.quantization = QUANTIZATION_FP8
    default_scaling_factor = torch.tensor([1], dtype=torch.float32)

    for layer in config.layers:
        if layer.attention:
            linear_layers = [
                layer.attention.dense,
            ]

            if isinstance(layer.attention.qkv, QKVConfig):
                linear_layers += [
                    layer.attention.qkv.q,
                    layer.attention.qkv.k,
                    layer.attention.qkv.v,
                ]
            elif isinstance(layer.attention.qkv, LinearConfig):
                linear_layers += [layer.attention.qkv]

        elif layer.recurrent:
            linear_layers = [
                layer.recurrent.linear_x,
                layer.recurrent.linear_y,
                layer.recurrent.linear_out,
            ]
        else:
            linear_layers = []

        moe_layers = []
        if isinstance(layer.mlp, MOEConfig):
            moe_layers.append(layer.mlp.experts.fc)
            moe_layers.append(layer.mlp.experts.proj)
        else:
            linear_layers.append(layer.mlp.fc)
            linear_layers.append(layer.mlp.proj)
            linear_layers.append(layer.mlp.gate)

        for linear_layer in linear_layers:
            if linear_layer:
                linear_layer.activation_scaling_factor = default_scaling_factor
                linear_layer.weights_scaling_factor = default_scaling_factor

        for moe_layer in moe_layers:
            if moe_layer:
                num_expert = moe_layer.weight.shape[0]
                # The moe plugin only supports a single activation scaling factor for all experts
                moe_layer.activation_scaling_factor = default_scaling_factor
                moe_layer.weights_scaling_factor = default_scaling_factor.expand(num_expert, 1)

        if layer.attention:
            layer.attention.kv_cache_dtype = KV_CACHE_FP8
            layer.attention.kv_cache_scaling_factor = torch.tensor([1.0], dtype=torch.float)
