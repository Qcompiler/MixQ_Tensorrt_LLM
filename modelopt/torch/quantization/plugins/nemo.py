# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Customization for Nemo Megatron GPT."""
from functools import partial
from types import ModuleType
from typing import Callable, Iterator, Tuple

import torch

# New nemo version should depend on megatron.core
from nemo.collections.nlp.modules.common.megatron.attention import CoreAttention

from ..nn import QuantInputBase, QuantModuleRegistry, TensorQuantizer
from .custom import _QuantFunctionalMixin

__all__ = []


def _quantized_bmm(self, input, mat2, *args, **kwargs):
    """Quantized version of BMM2 in nemo CoreAttention."""
    attn, v = input, mat2
    return torch._bmm(attn, self.v_bmm_quantizer(v), *args, **kwargs)


def _quantized_baddbmm(self, input, batch1, batch2, *args, **kwargs):
    """Quantized version of BMM1 in nemo CoreAttention."""
    q, k = batch1, batch2
    return torch._baddbmm(input, self.q_bmm_quantizer(q), self.k_bmm_quantizer(k), *args, **kwargs)


class _QuantCoreAttention(_QuantFunctionalMixin):
    """Quantized base class for CoreAttention."""

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


QuantModuleRegistry.register({CoreAttention: "nemo_core_attention"})(_QuantCoreAttention)
