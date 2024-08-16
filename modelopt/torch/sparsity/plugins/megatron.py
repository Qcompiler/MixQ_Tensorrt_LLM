# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


"""Support megatron parallel linear."""

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

from modelopt.torch.sparsity.config import SparseGPTConfig, SparseMagnitudeConfig
from modelopt.torch.sparsity.module import SparseModule, SpDMRegistry


class _MegatronParallelLinear(SparseModule):
    def _get_shard_axis_dict(self):
        raise NotImplementedError

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets)

        sparse_state_dict, sharded_axis_dict = {}, self._get_shard_axis_dict()
        for k, v in self.state_dict(prefix="", keep_vars=True).items():
            if k == "_weight_mask":
                sparse_state_dict[k] = v
        if sparse_state_dict:
            sharded_state_dict.update(
                **make_sharded_tensors_for_checkpoint(
                    sparse_state_dict, prefix, sharded_axis_dict, sharded_offsets
                )
            )

        return sharded_state_dict


@SpDMRegistry.register(
    {ColumnParallelLinear: "megatron.core.tensor_parallel.layers.ColumnParallelLinear"}
)
class _MegatronColumnParallelLinear(_MegatronParallelLinear):
    def _get_shard_axis_dict(self):
        return {"_weight_mask": 0}


@SpDMRegistry.register(
    {RowParallelLinear: "megatron.core.tensor_parallel.layers.RowParallelLinear"}
)
class _MegatronRowParallelLinear(_MegatronParallelLinear):
    def _get_shard_axis_dict(self):
        return {"_weight_mask": 1}


def _get_extra_rules():
    """Get the extra rules for megatron."""
    return {
        "megatron.core.tensor_parallel.layers.ColumnParallelLinear": {
            "*": {},
            "*output_layer*": None,
        },
        "megatron.core.tensor_parallel.layers.RowParallelLinear": {
            "*": {},
            "*output_layer*": None,
        },
    }


# Update the default rules
SparseMagnitudeConfig.register_default(_get_extra_rules())
SparseGPTConfig.register_default(_get_extra_rules())
