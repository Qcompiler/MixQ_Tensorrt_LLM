# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Support quantization for apex linear layers."""


from functools import partial

import apex.transformer.tensor_parallel.layers as apex_parallel

from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.quantization.nn.modules.quant_linear import _QuantLinear

from ..nn import QuantModuleRegistry
from .custom import _ParallelLinear


@QuantModuleRegistry.register(
    {
        apex_parallel.ColumnParallelLinear: "apex_ColumnParallelLinear",
        apex_parallel.RowParallelLinear: "apex_RowParallelLinear",
    }
)
class _ApexParallelLinear(DynamicModule):
    def _setup(self):
        quantized_linear_fn = partial(
            _QuantLinear.quantized_linear_fn,
            apex_parallel,
            "linear_with_grad_accumulation_and_async_allreduce",
            self,
        )
        self._forward_impl = quantized_linear_fn
        _ParallelLinear._setup(self)
