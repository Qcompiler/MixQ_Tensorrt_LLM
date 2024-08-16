# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utility to convert a Model Optimizer exported model to vLLM Checkpoint."""

import tempfile
from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn as nn


def _convert_weights_to_fp8(
    state_dict: Dict[str, torch.tensor], weights_to_convert: List[str]
) -> Dict[str, torch.tensor]:
    """Converts the original weights to FP8E4M3 from FP16."""
    for weight_name in weights_to_convert:
        weight_scale_name = weight_name + "_scale"
        if weight_scale_name not in state_dict.keys():
            continue
        loaded_weight = state_dict[weight_name]
        scale = state_dict[weight_scale_name]
        state_dict[weight_name] = (loaded_weight.cpu() / scale.cpu()).to(torch.float8_e4m3fn)
    return state_dict


def _convert_scales_for_vllm(key, value):
    """Replaces the names of *quantizer._amax to _scale."""
    replacements = {
        "weight_quantizer._amax": "weight_scale",
        "input_quantizer._amax": "act_scale",
    }
    for old_suffix, new_suffix in replacements.items():
        if key.endswith(old_suffix):
            new_key = key[: len(key) - len(old_suffix)] + new_suffix
            new_value = value / 448
            return new_key, new_value

    return key, value


def _convert_to_vllm_compatible_weights(input_state_dict: Dict[str, torch.tensor]):
    """Util function to modify the modelopt state dict to vLLM checkpoint."""
    weights_to_convert = []
    vllm_state_dict = {}
    for key, value in input_state_dict.items():
        if key.endswith("_amax"):
            new_key, new_value = _convert_scales_for_vllm(key, value)
            # Only add if the replacement happened.
            if key != new_key:
                vllm_state_dict[new_key] = new_value
        else:
            weights_to_convert.append(key)
            vllm_state_dict[key] = value
    # Conversion can only happen after all the amax values are read.
    vllm_state_dict = _convert_weights_to_fp8(vllm_state_dict, weights_to_convert)
    return vllm_state_dict


def _is_fp8(model):
    for _, layer in model.named_modules():
        if model == layer:
            continue

        if isinstance(layer, nn.Module):
            if "TensorQuantizer" in type(layer).__name__ and layer.is_enabled:
                return layer.num_bits == (4, 3)

            return_value = _is_fp8(layer)
            if return_value is not None:
                return return_value

    return None


def export_to_vllm(
    model: nn.Module, tokenizer: nn.Module, export_path: Union[Path, str] = tempfile.gettempdir()
):
    """Exports the torch model to vLLM checkpoint and saves to export_dir.

    Args:
        model: the torch model
        tokenizer: the tokenizer used for model
        export_path: Path for exporting the vLLM compatible quantized checkpoint

    """
    assert _is_fp8(model), "Only supports FP8 VLLM export."
    vllm_state_dict = _convert_to_vllm_compatible_weights(model.state_dict())

    # create directory
    Path(export_path).mkdir(parents=True, exist_ok=True)

    # save the state dict
    torch.save(vllm_state_dict, Path(export_path, "model.pt"))

    # save the config.json
    config_path = Path(export_path, "config.json")
    setattr(
        model.config, "quantization_config", {"quant_method": "fp8", "activation_scheme": "static"}
    )
    model.config.to_json_file(config_path)

    # save the tokenizer
    tokenizer.save_pretrained(Path(export_path))
