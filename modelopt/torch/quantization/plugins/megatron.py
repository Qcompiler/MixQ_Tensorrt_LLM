# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Support quantization for megatron linear layers."""


import megatron.core.tensor_parallel.layers as megatron_parallel
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

from ..config import QuantizerAttributeConfig
from ..nn import QuantModuleRegistry
from .custom import _ParallelLinear

__all__ = []


class _MegatronParallelLinear(_ParallelLinear):
    _functionals_to_replace = [
        (megatron_parallel, "linear_with_grad_accumulation_and_async_allreduce"),
        (megatron_parallel, "linear_with_frozen_weight"),
    ]

    def _process_weight_quantizer_amax(self, k, v, quantizer_state_dict):
        if v.ndim == 0:
            quantizer_state_dict[k] = v.view(-1)
        elif v.ndim == 2:
            quantizer_state_dict[k] = v.view(self.weight.shape[0], -1)
        else:
            raise ValueError(f"Invalid weight quantizer {k} amax: {v}, {v.shape}")

    def _process_activation_quantizer_amax(self, k, v, quantizer_state_dict):
        assert v.ndim == 0, f"Invalid activation quantizer amax: {v}, {v.shape}"
        quantizer_state_dict[k] = v.view(-1)

    def _process_activation_quantizer_pre_quant_scale(self, k, v, quantizer_state_dict):
        assert v.ndim == 1, f"Invalid activation quantizer pre_quant_scale: {v}, {v.shape}"
        quantizer_state_dict[k] = v

    def _get_shard_axis_dict(self):
        raise NotImplementedError

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets)

        quantizer_state_dict, sharded_axis_dict = {}, self._get_shard_axis_dict()
        for k, v in self.state_dict(prefix="", keep_vars=True).items():
            if "weight_quantizer." in k:
                assert k.endswith("._amax"), f"Invalid weight quantizer state: {k}"
                self._process_weight_quantizer_amax(k, v, quantizer_state_dict)
            elif ("input_quantizer" in k or "output_quantizer" in k) and k.endswith("._amax"):
                self._process_activation_quantizer_amax(k, v, quantizer_state_dict)
            elif k.endswith("input_quantizer._pre_quant_scale"):
                self._process_activation_quantizer_pre_quant_scale(k, v, quantizer_state_dict)

        sharded_state_dict.update(
            **make_sharded_tensors_for_checkpoint(
                quantizer_state_dict, prefix, sharded_axis_dict, sharded_offsets
            )
        )
        return sharded_state_dict


@QuantModuleRegistry.register(
    {megatron_parallel.ColumnParallelLinear: "megatron_ColumnParallelLinear"}
)
class _MegatronColumnParallelLinear(_MegatronParallelLinear):
    def _get_shard_axis_dict(self):
        shard_axis_dict = {}
        for k, v in self.state_dict(prefix="", keep_vars=True).items():
            if "weight_quantizer." in k and v.ndim != 0:
                shard_axis_dict[k] = 0
        return shard_axis_dict


@QuantModuleRegistry.register({megatron_parallel.RowParallelLinear: "megatron_RowParallelLinear"})
class _MegatronRowParallelLinear(_MegatronParallelLinear):
    def _get_shard_axis_dict(self):
        shard_axis_dict = {}
        for k, v in self.state_dict(prefix="", keep_vars=True).items():
            if "weight_quantizer" in k:
                assert "._amax" in k, f"Invalid weight quantizer state: {k}"
                submodule_name = k.split("weight_quantizer")[-1].split("._amax")[0].split(".")[-1]
                quantizer = self.weight_quantizer.get_submodule(submodule_name)
                # The weights are split across dim -1; Only static block quantization requires sharding
                if quantizer.block_sizes and quantizer.block_sizes.get("type", None) != "dynamic":
                    assert (-1 in quantizer.block_sizes or 1 in quantizer.block_sizes) and len(
                        QuantizerAttributeConfig._get_block_quant_axes_and_sizes(
                            quantizer.block_sizes
                        )
                    ) == 1, f"Invalid block sizes: {quantizer.block_sizes}"
                    shard_axis_dict[k] = 1
            elif "input_quantizer._pre_quant_scale" in k:
                shard_axis_dict[k] = 0
        return shard_axis_dict
