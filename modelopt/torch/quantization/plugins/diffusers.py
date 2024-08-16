# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Support quantization of diffusers layers."""
from functools import partial
from types import ModuleType
from typing import Callable, Iterator, Tuple

import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear

from modelopt.torch.quantization.nn import (
    QuantConv2d,
    QuantInputBase,
    QuantLinear,
    QuantLinearConvBase,
    QuantModuleRegistry,
    TensorQuantizer,
)

from .custom import _QuantFunctionalMixin


class _QuantLoRACompatibleLinearConvBase(QuantLinearConvBase):
    def _setup(self):
        assert self.lora_layer is None, (
            f"To quantize {self}, lora_layer should be None. Please fuse the LoRA layer before"
            " quantization."
        )
        return super()._setup()


@QuantModuleRegistry.register({LoRACompatibleConv: "LoRACompatibleConv"})
class _QuantLoRACompatibleConv(_QuantLoRACompatibleLinearConvBase):
    default_quant_desc_weight = QuantConv2d.default_quant_desc_weight


@QuantModuleRegistry.register({LoRACompatibleLinear: "LoRACompatibleLinear"})
class _QuantLoRACompatibleLinear(_QuantLoRACompatibleLinearConvBase):
    default_quant_desc_weight = QuantLinear.default_quant_desc_weight


def _quantized_bmm(self, input, mat2, *args, **kwargs):
    attn, v = input, mat2
    return torch._bmm(self.softmax_quantizer(attn), self.v_bmm_quantizer(v), *args, **kwargs)


def _quantized_baddbmm(self, input, batch1, batch2, *args, **kwargs):
    q, k = batch1, batch2
    return torch._baddbmm(input, self.q_bmm_quantizer(q), self.k_bmm_quantizer(k), *args, **kwargs)


class _QuantAttention(_QuantFunctionalMixin):
    """FP8 processor for performing attention-related computations."""

    _functionals_to_replace = [
        (torch, "bmm", _quantized_bmm),
        (torch, "baddbmm", _quantized_baddbmm),
    ]

    @property
    def functionals_to_replace(self) -> Iterator[Tuple[ModuleType, str, Callable]]:
        for package, func_name, quantized_func in self._functionals_to_replace:
            if not hasattr(package, func_name):
                continue
            quantized_func = partial(quantized_func, self)
            yield package, func_name, quantized_func

    def _setup(self):
        self.q_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.k_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.v_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.softmax_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)


QuantModuleRegistry.register({Attention: "Attention"})(_QuantAttention)
