# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utils for scaling factors adjustments."""


from typing import List

import torch


def get_weights_scaling_factor(weight, group_size):
    """Calculate the weight scaling facotrs for a given group size."""
    [n, k] = weight.shape

    if group_size != 0:
        # int4_awq
        if k % group_size != 0:
            raise NotImplementedError(
                "Weight shape is not divisible for block size for block quantiation."
            )
        weight = weight.reshape(n, k // group_size, group_size)
        maxbound = 7.0
    else:
        # int8_sq
        maxbound = 127.0
    amax = weight.abs().max(dim=-1)[0].float()

    weights_scaling_factor = amax / maxbound

    # Let's filter the zeros in the scaling factor if the weights are zero
    # to avoid the divided-by-zero error..
    weights_scaling_factor[weights_scaling_factor == 0] = 1.0

    return weights_scaling_factor


def resmooth_and_get_scale(
    merged_weights: torch.Tensor,
    pre_quant_scales: List[torch.Tensor],
    ranks: int,
    group_size: int,
    avg_pre_quant_scale: torch.Tensor = None,
):
    """Resmooths weights from a single or multiple ranks.

    Args:
        merged_weights: Merged weights from ranks.
        pre_quant_scales: List of pre-quantization scales for each rank.
        ranks: Number of ranks.
        group_size: Group size of the quantization block.
        avg_pre_quant_scale (optional): If not provided, weights will be resmoothed using
            the average of pre_quant_scales.

    Returns:
        weights: Resmoothed weights.
        weight_scaling_factors: Resmoothed scaling factors.
        avg_pre_quant_scale: Calculated average of the quantization scale.
    """
    if avg_pre_quant_scale is None:
        avg_pre_quant_scale = torch.stack(pre_quant_scales).mean(dim=0)

    assert (
        len(pre_quant_scales) > 0 and avg_pre_quant_scale.numel() == merged_weights.shape[1]
    ), "Shape of pre_quant_scales and weights do not match."
    weights = torch.chunk(merged_weights, ranks, dim=0)

    scales = []
    new_weights = []
    for i, p_scaling_factor in enumerate(pre_quant_scales):
        # De smooth & Re smooth
        weight = (
            weights[i]
            * p_scaling_factor.type(weights[i].dtype)
            / avg_pre_quant_scale.type(weights[i].dtype)
        )
        new_weights.append(weight)
        scale = get_weights_scaling_factor(weight, group_size)
        scales.append(scale)

    return torch.cat(new_weights, dim=0), torch.cat(scales, dim=0), avg_pre_quant_scale


def adjust_attn_amax_values(module):
    """Adjusts the amax values for the attention layers."""
    projection_prefixes = ["q", "k", "v"]
    max_amax = float("-inf")
    proj_layers = []

    # Find all projection layers whose names contain 'q', 'k', or 'v'
    for name, sub_module in module.named_children():
        for prefix in projection_prefixes:
            if (
                prefix in name
                and hasattr(sub_module, "weight_quantizer")
                and hasattr(sub_module.weight_quantizer, "amax")
            ):
                proj_layers.append(sub_module)
                max_amax = max(max_amax, sub_module.weight_quantizer.amax.item())

    if not proj_layers:
        raise ValueError(
            "No projection layers with the specified prefixes ('q', 'k', 'v') have amax attributes"
        )

    assert max_amax > 0, "max_amax must be positive."

    # Set all amax values to the maximum found
    for proj_layer in proj_layers:
        proj_layer.weight_quantizer.amax.fill_(max_amax)
