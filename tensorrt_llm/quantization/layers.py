# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Optional

import numpy as np
import tensorrt as trt

from .._common import default_net, precision
from .._utils import fp32_array, is_same_dtype
from ..functional import (ACT2FN, AllReduceFusionOp, AllReduceFusionParams,
                          AttentionMaskType, PositionEmbeddingType,
                          RotaryScalingType, Tensor, allgather, allreduce, cast,
                          concat, constant, embedding, generate_alibi_slopes,
                          gpt_attention, matmul, mul, shape, slice, softmax,
                          split, where)
from ..layers import SpecDecodingParams
from ..layers.embedding import Embedding
from ..layers.linear import Linear, RowLinear
from ..module import Module
from ..parameter import Parameter
from .functional import (dequantize, fp8_rowwise_gemm, fp8_rowwise_rms_norm,
                         quantize, quantize_fp8_per_token, quantize_per_token,
                         quantize_tensor, smooth_quant_gemm,
                         smooth_quant_layer_norm, smooth_quant_rms_norm,
                         weight_only_groupwise_quant_matmul,
                         weight_only_quant_matmul)
from .mode import QuantMode


class Quantize(Module):
    """
        Quantize Layer
        For per-tensor mode, the scaling factor is a scalar.
        For per-channel mode, the scaling factor is a vector.
        """

    def __init__(
        self,
        output_dtype: str = 'int8',
        scaling_factor_dtype: str = 'float32',
        in_channels: int = -1,
        axis=-1,
    ) -> None:
        super().__init__()
        self.scaling_factor = Parameter(shape=(in_channels, ) if axis != -1 else
                                        (),
                                        dtype=scaling_factor_dtype)
        self.output_dtype = output_dtype
        self.axis = axis

    def forward(self, x):
        return quantize(x, self.scaling_factor.value, self.output_dtype,
                        self.axis)


class QuantizePerToken(Module):
    """
        Quantize Per Token and compute dynamic scales for SmoothQuant
        """

    def forward(self, x):
        return quantize_per_token(x)


class Dequantize(Module):
    """
        Dequantize Layer.
        """

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.scaling_factor = Parameter(shape=(), dtype='float32')
        self.axis = axis

    def forward(self, input):
        return dequantize(input, self.scaling_factor.value, self.axis)


class SmoothQuantLinear(Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 gather_output=True,
                 quant_mode=QuantMode(0)):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size,
                         gather_output=gather_output)

        if not quant_mode.has_act_and_weight_quant():
            raise ValueError(
                "SmoothQuant Linear has to have act+weight quantization mode set"
            )

        weights_dtype = dtype
        if quant_mode.has_act_and_weight_quant():
            weights_dtype = "int8"

        self.weight = Parameter(shape=(self.out_features, self.in_features),
                                dtype=weights_dtype)

        if quant_mode.has_act_and_weight_quant():
            scale_shape = (1, self.out_features
                           ) if quant_mode.has_per_channel_scaling() else (1, 1)
            self.per_channel_scale = Parameter(shape=scale_shape,
                                               dtype="float32")

        if quant_mode.has_act_static_scaling():
            self.act_scale = Parameter(shape=(1, 1), dtype="float32")

        self.quant_mode = quant_mode

    def forward(self, x, lora_runtime_params=None):
        assert lora_runtime_params is None, "lora is not supported on SmoothQuantLinear now"
        if self.quant_mode.has_act_static_scaling():
            per_token_scale = self.act_scale.value
        else:
            # If we are in SmoothQuant with dynamic activation scaling,
            # input x has to be a tuple of int8 tensor and fp32 scaling factors
            x, per_token_scale = x
        x = smooth_quant_gemm(x, self.weight.value, per_token_scale,
                              self.per_channel_scale.value,
                              self.quant_mode.has_per_token_dynamic_scaling(),
                              self.quant_mode.has_per_channel_scaling())

        if self.bias is not None:
            x = x + self.bias.value

        if self.gather_output and self.tp_size > 1 and self.tp_group is not None:
            # [dim0, local_dim] -> [dim0 * tp_size, local_dim] --> [dim0, local_dim * tp_size]
            x = allgather(x, self.tp_group, gather_dim=1)

        return x


SmoothQuantColumnLinear = SmoothQuantLinear


class SmoothQuantRowLinear(RowLinear):

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            dtype=None,
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
    ):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size)
        if not quant_mode.has_act_and_weight_quant():
            raise ValueError(
                "SmoothQuant Linear has to have act+weight quantization mode set"
            )
        weights_dtype = dtype
        if quant_mode.has_act_and_weight_quant():
            weights_dtype = "int8"

        self.weight = Parameter(shape=(self.out_features, self.in_features),
                                dtype=weights_dtype)
        self.smoother = Parameter(shape=(1, self.in_features), dtype="float32")
        if quant_mode.has_act_and_weight_quant():
            scale_shape = (1, self.out_features
                           ) if quant_mode.has_per_channel_scaling() else (1, 1)
            self.per_channel_scale = Parameter(shape=scale_shape,
                                               dtype="float32")

        if quant_mode.has_act_static_scaling():
            self.act_scale = Parameter(shape=(1, 1), dtype="float32")

        self.quant_mode = quant_mode

    def forward(self, x, lora_runtime_params=None, reduce_fusion_params=None):
        assert lora_runtime_params is None, "lora is not supported on SmoothQuantRowLinear now"
        if self.quant_mode.has_act_static_scaling():
            per_token_scale = self.act_scale.value
        else:
            x, per_token_scale = x
        x = smooth_quant_gemm(x, self.weight.value, per_token_scale,
                              self.per_channel_scale.value,
                              self.quant_mode.has_per_token_dynamic_scaling(),
                              self.quant_mode.has_per_channel_scaling())

        if self.tp_size > 1 and self.tp_group is not None:
            need_bias = self.bias is not None
            fuse_bias_into_all_reduce = need_bias and (
                reduce_fusion_params
                is not None) and (reduce_fusion_params.fusion_op
                                  == AllReduceFusionOp.RESIDUAL_RMS_NORM)
            if fuse_bias_into_all_reduce:
                reduce_fusion_params.bias = self.bias.value
            x = allreduce(x,
                          self.tp_group,
                          reduce_fusion_params=reduce_fusion_params)
            if need_bias and not fuse_bias_into_all_reduce:
                x = x + self.bias.value
            return x

        if self.bias is not None:
            x = x + self.bias.value

        return x


class SmoothQuantLayerNorm(Module):

    def __init__(
            self,
            normalized_shape,
            eps=1e-05,
            elementwise_affine=True,
            bias=True,
            dtype=None,
            quant_mode=QuantMode(0),
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        if not quant_mode.has_act_and_weight_quant():
            raise ValueError(
                "SmoothQuant layer norm has to have some quantization mode set")
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(shape=self.normalized_shape, dtype=dtype)
            if bias:
                self.bias = Parameter(shape=self.normalized_shape, dtype=dtype)
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.eps = eps
        self.dtype = dtype
        self.quant_mode = quant_mode

        if self.quant_mode.has_act_and_weight_quant():
            self.scale_to_int = Parameter(shape=(1, ), dtype=dtype)
        else:
            self.register_parameter('scale_to_int', None)

    def forward(self, x):
        weight = None if self.weight is None else self.weight.value
        bias = None if self.bias is None else self.bias.value
        scale = None if self.scale_to_int is None else self.scale_to_int.value
        return smooth_quant_layer_norm(
            x,
            self.normalized_shape,
            weight,
            bias,
            scale,
            self.eps,
            dynamic_act_scaling=self.quant_mode.has_per_token_dynamic_scaling())


class SmoothQuantRmsNorm(Module):

    def __init__(
            self,
            normalized_shape,
            eps=1e-06,
            elementwise_affine=True,
            dtype=None,
            quant_mode=QuantMode(0),
            bias=False,
            clamp_val=None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        if not quant_mode.has_act_and_weight_quant():
            raise ValueError(
                "SmoothQuant Rms norm has to have some quantization mode set")
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('bias', None)
        if clamp_val:
            if not (isinstance(clamp_val, list) and len(clamp_val) == 2):
                raise ValueError(f'unsupported clamp_val {clamp_val}')
            self.clamp_val = Parameter(np.array(clamp_val, dtype=np.float32),
                                       dtype='float32',
                                       is_buffer=True)
        else:
            self.register_parameter('clamp_val', None)

        self.eps = eps
        self.dtype = dtype
        self.quant_mode = quant_mode

        if self.quant_mode.has_act_and_weight_quant():
            self.scale_to_int = Parameter(shape=(1, ), dtype=dtype)
        else:
            self.register_parameter('scale_to_int', None)

    def forward(self, x):
        weight = None if self.weight is None else self.weight.value
        bias = None if self.bias is None else self.bias.value
        scale = None if self.scale_to_int is None else self.scale_to_int.value
        clamp_val = None if self.clamp_val is None else self.clamp_val.value
        return smooth_quant_rms_norm(
            x,
            self.normalized_shape,
            weight,
            bias,
            scale,
            clamp_val,
            self.eps,
            dynamic_act_scaling=self.quant_mode.has_per_token_dynamic_scaling())


class Fp8RowwiseRmsNorm(Module):

    def __init__(
            self,
            normalized_shape,
            eps=1e-06,
            elementwise_affine=True,
            dtype=None,
            quant_mode=QuantMode(0),
            bias=False,
            clamp_val=None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        if not quant_mode.has_fp8_rowwise():
            raise ValueError(
                "Fp8 Rowwise Rms norm has to have some quantization mode set")
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('bias', None)
        if clamp_val:
            if not (isinstance(clamp_val, list) and len(clamp_val) == 2):
                raise ValueError(f'unsupported clamp_val {clamp_val}')
            self.clamp_val = Parameter(np.array(clamp_val, dtype=np.float32),
                                       dtype='float32',
                                       is_buffer=True)
        else:
            self.register_parameter('clamp_val', None)
        self.register_parameter('scale_to_int', None)

        self.eps = eps
        self.dtype = dtype
        self.quant_mode = quant_mode

    def forward(self, x):
        weight = None if self.weight is None else self.weight.value
        bias = None if self.bias is None else self.bias.value
        scale = None if self.scale_to_int is None else self.scale_to_int.value
        return fp8_rowwise_rms_norm(
            x,
            self.normalized_shape,
            weight,
            bias,
            scale,
            self.clamp_val.value,
            self.eps,
            dynamic_act_scaling=self.quant_mode.has_fp8_rowwise())


class Fp8RowwiseLinear(Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 gather_output=True,
                 quant_mode=QuantMode(0)):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size,
                         gather_output=gather_output)

        if not quant_mode.has_fp8_rowwise():
            raise ValueError(
                "Fp8 Rowwise Linear has to have act+weight quantization mode set"
            )

        weights_dtype = dtype
        if quant_mode.has_fp8_rowwise():
            weights_dtype = "fp8"

        self.weight = Parameter(shape=(self.out_features, self.in_features),
                                dtype=weights_dtype)

        if quant_mode.has_fp8_rowwise():
            scale_shape = (
                self.out_features, ) if quant_mode.has_fp8_rowwise() else (1, )
            self.per_channel_scale = Parameter(shape=scale_shape,
                                               dtype="float32")

        self.quant_mode = quant_mode

    def forward(self, x, lora_runtime_params=None):
        assert lora_runtime_params is None, "lora is not supported on SmoothQuantLinear now"
        x, per_token_scale = x
        x = fp8_rowwise_gemm(x, self.weight.value, per_token_scale,
                             self.per_channel_scale.value,
                             self.quant_mode.has_per_token_dynamic_scaling(),
                             self.quant_mode.has_per_channel_scaling())

        if self.bias is not None:
            x = x + self.bias.value

        if self.gather_output and self.tp_size > 1 and self.tp_group is not None:
            # [dim0, local_dim] -> [dim0 * tp_size, local_dim] --> [dim0, local_dim * tp_size]
            x = allgather(x, self.tp_group, gather_dim=1)

        return x


Fp8RowwiseColumnLinear = Fp8RowwiseLinear


class Fp8RowwiseRowLinear(RowLinear):

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            dtype=None,
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
    ):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size)
        if not quant_mode.has_fp8_rowwise():
            raise ValueError(
                "Fp8 Rowwise Linear has to have act+weight quantization mode set"
            )
        weights_dtype = dtype
        if quant_mode.has_fp8_rowwise():
            weights_dtype = "fp8"

        self.weight = Parameter(shape=(self.out_features, self.in_features),
                                dtype=weights_dtype)
        if quant_mode.has_fp8_rowwise():
            scale_shape = (
                self.out_features, ) if quant_mode.has_fp8_rowwise() else (1, )
            self.per_channel_scale = Parameter(shape=scale_shape,
                                               dtype="float32")

        self.quant_mode = quant_mode

    def forward(self, x, lora_runtime_params=None, reduce_fusion_params=None):
        assert lora_runtime_params is None, "lora is not supported on SmoothQuantRowLinear now"
        x, per_token_scale = x
        x = fp8_rowwise_gemm(x, self.weight.value, per_token_scale,
                             self.per_channel_scale.value,
                             self.quant_mode.has_fp8_rowwise(),
                             self.quant_mode.has_fp8_rowwise())

        if self.tp_size > 1 and self.tp_group is not None:
            need_bias = self.bias is not None
            fuse_bias_into_all_reduce = need_bias and (
                reduce_fusion_params
                is not None) and (reduce_fusion_params.fusion_op
                                  == AllReduceFusionOp.RESIDUAL_RMS_NORM)
            if fuse_bias_into_all_reduce:
                reduce_fusion_params.bias = self.bias.value
            x = allreduce(x,
                          self.tp_group,
                          reduce_fusion_params=reduce_fusion_params)
            if need_bias and not fuse_bias_into_all_reduce:
                x = x + self.bias.value
            return x

        if self.bias is not None:
            x = x + self.bias.value

        return x


class WeightOnlyQuantLinear(Linear):

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            dtype=None,
            tp_group=None,
            tp_size=1,
            gather_output=True,
            quant_mode=QuantMode.use_weight_only(),
            transa=False,
            transb=False,
    ):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size,
                         gather_output=gather_output)
        if quant_mode.is_int8_weight_only():
            self.weight_only_quant_mode = 1
            quant_type_size_in_bits = 8
        elif quant_mode.is_int4_weight_only():
            self.weight_only_quant_mode = 2
            quant_type_size_in_bits = 4
        # we use a fake tensor with data_type = int8
        self.weight = Parameter(shape=(self.in_features,
                                       int(self.out_features *
                                           quant_type_size_in_bits / 8)),
                                dtype="int8")

        scale_shape = (self.out_features, )
        self.per_channel_scale = Parameter(shape=scale_shape, dtype=dtype)

        self.transa = transa
        self.transb = transb

    def forward(self, x, lora_runtime_params=None):
        # ootb has not supported int4 yet.
        if self.weight_only_quant_mode == 2 and not default_net(
        ).plugin_config.weight_only_quant_matmul_plugin:
            raise TypeError(
                "Int4 Weight-only Quant MatMul is only supported with plugin")
        hidden_state = x
        x = weight_only_quant_matmul(x, self.weight.value,
                                     self.per_channel_scale.value,
                                     self.weight_only_quant_mode, self.dtype,
                                     self.transa, self.transb)

        if default_net(
        ).plugin_config.lora_plugin and lora_runtime_params is not None:
            x = x + self.lora(hidden_state,
                              lora_runtime_params=lora_runtime_params)

        if self.bias is not None:
            x = x + self.bias.value

        if self.gather_output and self.tp_size > 1 and self.tp_group is not None:
            # [dim0, local_dim] -> [dim0 * tp_size, local_dim] --> [dim0, local_dim * tp_size]
            x = allgather(x, self.tp_group, gather_dim=1)

        return x


WeightOnlyQuantColumnLinear = WeightOnlyQuantLinear


class WeightOnlyQuantRowLinear(RowLinear):

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            dtype=None,
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode.use_weight_only(),
    ):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size)
        if quant_mode.is_int8_weight_only():
            self.weight_only_quant_mode = 1
        elif quant_mode.is_int4_weight_only():
            self.weight_only_quant_mode = 2
        #we use a fake tensor with data_type = int8
        self.weight = Parameter(shape=(self.in_features,
                                       int(self.out_features /
                                           self.weight_only_quant_mode)),
                                dtype="int8")
        self.per_channel_scale = Parameter(shape=(self.out_features, ),
                                           dtype=dtype)

    def forward(self, x, lora_runtime_params=None, reduce_fusion_params=None):
        hidden_state = x
        x = weight_only_quant_matmul(x, self.weight.value,
                                     self.per_channel_scale.value,
                                     self.weight_only_quant_mode, self.dtype)

        if default_net(
        ).plugin_config.lora_plugin and lora_runtime_params is not None:
            x = x + self.lora(hidden_state,
                              lora_runtime_params=lora_runtime_params)

        if self.tp_size > 1 and self.tp_group is not None:
            need_bias = self.bias is not None
            fuse_bias_into_all_reduce = need_bias and (
                reduce_fusion_params
                is not None) and (reduce_fusion_params.fusion_op
                                  == AllReduceFusionOp.RESIDUAL_RMS_NORM)
            if fuse_bias_into_all_reduce:
                reduce_fusion_params.bias = self.bias.value
            x = allreduce(x,
                          self.tp_group,
                          reduce_fusion_params=reduce_fusion_params)
            if need_bias and not fuse_bias_into_all_reduce:
                x = x + self.bias.value
            return x

        if self.bias is not None:
            x = x + self.bias.value

        return x


class WeightOnlyQuantEmbedding(Embedding):

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            dtype: Optional[str] = None,
            tp_size: int = 1,
            tp_group: Optional[list] = None,
            sharding_dim: int = 0,
            tp_rank: Optional[int] = None,
            quant_mode=QuantMode.use_weight_only(),
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            dtype,  # dtype,
            tp_size,
            tp_group,
            sharding_dim,
            tp_rank)
        # only support int8 wo now
        # TODO support int4 wo
        self.quant_mode = quant_mode
        self.per_token_scale = Parameter(shape=(self.num_embeddings, ),
                                         dtype=dtype)

        if sharding_dim == 1:
            self.weight = Parameter(shape=(self.num_embeddings,
                                           self.embedding_dim // self.tp_size),
                                    dtype="int8")
        elif sharding_dim == 0:
            self.weight = Parameter(shape=(math.ceil(
                self.num_embeddings / self.tp_size), self.embedding_dim),
                                    dtype="int8")

    def forward(self, x):
        result = embedding(x,
                           self.weight.value,
                           tp_size=self.tp_size,
                           tp_group=self.tp_group,
                           sharding_dim=self.sharding_dim,
                           tp_rank=self.tp_rank,
                           per_token_scale=self.per_token_scale.value)

        return result


class WeightOnlyGroupwiseQuantLinear(Linear):

    def __init__(
        self,
        in_features,
        out_features,
        group_size=128,
        pre_quant_scale=False,
        zero=False,
        bias=False,
        dtype=None,
        tp_group=None,
        tp_size=1,
        gather_output=True,
        use_w4a8_awq=False,
    ):

        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size,
                         gather_output=gather_output)

        # Flags for indicating whether the corresponding inputs are applied in quant_algo
        BIAS = 1
        ZERO = 2
        PRE_QUANT_SCALE = 4
        W4A8_AWQ = 8

        self.quant_algo = use_w4a8_awq * W4A8_AWQ + pre_quant_scale * PRE_QUANT_SCALE + zero * ZERO + bias * BIAS
        self.group_size = group_size
        self.weight = Parameter(shape=(self.in_features,
                                       self.out_features // 4),
                                dtype=dtype)

        scale_shape = (self.in_features // group_size, self.out_features)
        self.weights_scaling_factor = Parameter(shape=scale_shape, dtype=dtype)

        if pre_quant_scale:
            self.prequant_scaling_factor = Parameter(shape=(1,
                                                            self.in_features),
                                                     dtype=dtype)
        else:
            self.register_parameter('prequant_scaling_factor', None)

        if zero:
            self.zero = Parameter(shape=scale_shape, dtype=dtype)
        else:
            self.register_parameter('zero', None)

        if use_w4a8_awq:
            self.alpha = Parameter(shape=(1, ), dtype="float32")
        else:
            self.register_parameter('alpha', None)

    def forward(self, x, lora_runtime_params=None):
        pre_quant_scale = self.prequant_scaling_factor.value if self.prequant_scaling_factor else None
        zero = self.zero.value if self.zero else None
        bias = self.bias.value if self.bias else None
        alpha = self.alpha.value if self.alpha else None

        hidden_state = x
        x = weight_only_groupwise_quant_matmul(
            x, pre_quant_scale, self.weight.value,
            self.weights_scaling_factor.value, zero, bias, alpha,
            self.quant_algo, self.group_size, self.dtype)

        if default_net(
        ).plugin_config.lora_plugin and lora_runtime_params is not None:
            x = x + self.lora(hidden_state,
                              lora_runtime_params=lora_runtime_params)

        if self.gather_output and self.tp_size > 1 and self.tp_group is not None:
            # [dim0, local_dim] -> [dim0 * tp_size, local_dim] --> [dim0, local_dim * tp_size]
            x = allgather(x, self.tp_group, gather_dim=1)

        return x


WeightOnlyGroupwiseQuantColumnLinear = WeightOnlyGroupwiseQuantLinear


class WeightOnlyGroupwiseQuantRowLinear(RowLinear):

    def __init__(
        self,
        in_features,
        out_features,
        group_size=128,
        pre_quant_scale=False,
        zero=False,
        bias=False,
        dtype=None,
        tp_group=None,
        tp_size=1,
        use_w4a8_awq=False,
    ):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size)

        # Flags for indicating whether the corresponding inputs are applied in quant_algo
        BIAS = 1
        ZERO = 2
        PRE_QUANT_SCALE = 4
        W4A8_AWQ = 8

        self.quant_algo = use_w4a8_awq * W4A8_AWQ + pre_quant_scale * PRE_QUANT_SCALE + zero * ZERO + bias * BIAS
        self.group_size = group_size
        self.weight = Parameter(shape=(self.in_features,
                                       self.out_features // 4),
                                dtype=dtype)

        scale_shape = (self.in_features // group_size, self.out_features)
        self.weights_scaling_factor = Parameter(shape=scale_shape, dtype=dtype)

        if pre_quant_scale:
            self.prequant_scaling_factor = Parameter(shape=(1,
                                                            self.in_features),
                                                     dtype=dtype)
        else:
            self.register_parameter('prequant_scaling_factor', None)

        if zero:
            self.zero = Parameter(shape=scale_shape, dtype=dtype)
        else:
            self.register_parameter('zero', None)

        if use_w4a8_awq:
            self.alpha = Parameter(shape=(1, ), dtype="float32")
        else:
            self.register_parameter('alpha', None)

    def forward(self, x, lora_runtime_params=None, reduce_fusion_params=None):
        pre_quant_scale = self.prequant_scaling_factor.value if self.prequant_scaling_factor else None
        zero = self.zero.value if self.zero else None
        bias = self.bias.value if self.bias else None
        alpha = self.alpha.value if self.alpha else None

        hidden_state = x
        x = weight_only_groupwise_quant_matmul(
            x, pre_quant_scale, self.weight.value,
            self.weights_scaling_factor.value, zero, bias, alpha,
            self.quant_algo, self.group_size, self.dtype)

        if default_net(
        ).plugin_config.lora_plugin and lora_runtime_params is not None:
            x = x + self.lora(hidden_state,
                              lora_runtime_params=lora_runtime_params)

        if self.tp_size > 1 and self.tp_group is not None:
            x = allreduce(x,
                          self.tp_group,
                          reduce_fusion_params=reduce_fusion_params)

        return x


class SmoothQuantMLP(Module):

    def __init__(
            self,
            hidden_size,
            ffn_hidden_size,
            hidden_act,
            bias=True,
            dtype=None,
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
    ):
        super().__init__()
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        fc_output_size = 2 * ffn_hidden_size if hidden_act == 'swiglu' else ffn_hidden_size
        self.fc = SmoothQuantColumnLinear(hidden_size,
                                          fc_output_size,
                                          bias=bias,
                                          dtype=dtype,
                                          tp_group=tp_group,
                                          tp_size=tp_size,
                                          gather_output=False,
                                          quant_mode=quant_mode)

        self.proj = SmoothQuantRowLinear(ffn_hidden_size,
                                         hidden_size,
                                         bias=bias,
                                         dtype=dtype,
                                         tp_group=tp_group,
                                         tp_size=tp_size,
                                         quant_mode=quant_mode)

        self.hidden_act = hidden_act
        self.quant_mode = quant_mode
        self.dtype = dtype

        if self.quant_mode.has_act_static_scaling():
            self.quantization_scaling_factor = Parameter(shape=(1, ),
                                                         dtype='float32')
        else:
            self.register_parameter('quantization_scaling_factor', None)

    def forward(self, hidden_states, lora_layer_params=None):

        inter = self.fc(hidden_states)
        inter = ACT2FN[self.hidden_act](inter)
        value = cast(self.proj.smoother.value, inter.dtype)
        inter = inter / value
        if self.quant_mode.has_act_and_weight_quant():
            if self.quant_mode.has_act_static_scaling():
                # Avoid quantization layers as it breaks int8 plugins
                inter = quantize_tensor(inter,
                                        self.quantization_scaling_factor.value)
            else:
                # Quantize per token outputs tuple:
                # quantized tensor and scaling factors per token
                inter = quantize_per_token(inter)
        output = self.proj(inter)
        return output


class Int8SmoothQuantRowLinear(RowLinear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size)
        self.activation_scaling_factor = Parameter(shape=(1, ),
                                                   dtype=trt.float32)
        self.weights_scaling_factor = Parameter(shape=(self.out_features, ),
                                                dtype=trt.float32)
        self.prequant_scaling_factor = Parameter(shape=(self.in_features, ),
                                                 dtype=dtype)
        self.weight = Parameter(shape=(self.out_features, self.in_features),
                                dtype=trt.int8)

    def forward(self, x, lora_runtime_params=None, reduce_fusion_params=None):
        lora_hidden_state = x if lora_runtime_params is not None else None
        if default_net().strongly_typed:
            assert is_same_dtype(
                x.dtype,
                self.dtype), f"Got input type {x.dtype}, expecting {self.dtype}"
        x = mul(x, self.prequant_scaling_factor.value)

        x = cast(x, self.activation_scaling_factor.value.dtype)

        quantized_out = quantize(x, self.activation_scaling_factor.value,
                                 'int8')

        dequantized_out = dequantize(quantized_out,
                                     self.activation_scaling_factor.value, -1,
                                     self.activation_scaling_factor.value.dtype)

        dequantized_out = cast(dequantized_out, self.dtype)

        w_deq_out = dequantize(self.weight.value,
                               self.weights_scaling_factor.value, 0,
                               self.weights_scaling_factor.value.dtype)

        w_deq_out = cast(w_deq_out, self.dtype)
        return self.multiply_reduce(dequantized_out,
                                    w_deq_out,
                                    gemm_plugin=None,
                                    reduce_fusion_params=reduce_fusion_params,
                                    lora_runtime_params=lora_runtime_params,
                                    lora_hidden_state=lora_hidden_state)


class Int8SmoothQuantLinear(Linear):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=None,
        tp_group=None,
        tp_size=1,
        gather_output=True,
    ):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size,
                         gather_output=gather_output)
        self.activation_scaling_factor = Parameter(shape=(1, ),
                                                   dtype=trt.float32)

        self.weights_scaling_factor = Parameter(shape=(self.out_features, ),
                                                dtype=trt.float32)
        self.prequant_scaling_factor = Parameter(shape=(self.in_features, ),
                                                 dtype=dtype)
        self.weight = Parameter(shape=(self.out_features, self.in_features),
                                dtype=trt.int8)

    def forward(self, x, lora_runtime_params=None):
        lora_hidden_state = x if lora_runtime_params is not None else None
        if default_net().strongly_typed:
            assert is_same_dtype(
                x.dtype,
                self.dtype), f"Got input type {x.dtype}, expecting {self.dtype}"
        x = mul(x, self.prequant_scaling_factor.value)
        x = cast(x, self.activation_scaling_factor.value.dtype)

        quantized_out = quantize(x, self.activation_scaling_factor.value,
                                 'int8')

        dequantized_out = dequantize(quantized_out,
                                     self.activation_scaling_factor.value, -1,
                                     self.activation_scaling_factor.value.dtype)

        dequantized_out = cast(dequantized_out, self.dtype)

        w_deq_out = dequantize(self.weight.value,
                               self.weights_scaling_factor.value, 0,
                               self.weights_scaling_factor.value.dtype)
        w_deq_out = cast(w_deq_out, self.dtype)

        return self.multiply_gather(dequantized_out,
                                    w_deq_out,
                                    gemm_plugin=None,
                                    lora_runtime_params=lora_runtime_params,
                                    lora_hidden_state=lora_hidden_state)


class FP8Linear(Linear):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=None,
        tp_group=None,
        tp_size=1,
        gather_output=True,
    ):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size,
                         gather_output=gather_output)
        self.weight = Parameter(shape=(self.out_features, self.in_features),
                                dtype='fp8')
        self.activation_scaling_factor = Parameter(shape=(1, ),
                                                   dtype=trt.float32)
        self.weights_scaling_factor = Parameter(shape=(1, ), dtype=trt.float32)

    def forward(self, x, lora_runtime_params=None):
        assert lora_runtime_params is None or default_net(
        ).plugin_config.lora_plugin == self.dtype

        if default_net().strongly_typed:
            assert is_same_dtype(
                x.dtype,
                self.dtype), f"Got input type {x.dtype}, expecting {self.dtype}"

        alpha = self.weights_scaling_factor.raw_value * self.activation_scaling_factor.raw_value
        activation_scaling_factor = cast(self.activation_scaling_factor.value,
                                         self.dtype)
        if x.dtype != trt.fp8:
            quantized_out = quantize(x, activation_scaling_factor, 'fp8')
            lora_hidden_state = x if lora_runtime_params is not None else None
        else:
            quantized_out = x
            # TODO: add fp8 LoRA support
            lora_hidden_state = dequantize(
                x, activation_scaling_factor, -1,
                self.dtype) if lora_runtime_params is not None else None

        weights_scaling_factor = cast(self.weights_scaling_factor.value,
                                      self.dtype)
        if self.weight.value.dtype != trt.fp8:
            w_quant_out = quantize(self.weight.value, weights_scaling_factor,
                                   'fp8')
        else:
            w_quant_out = self.weight.value

        gemm_plugin = default_net().plugin_config.gemm_plugin
        if gemm_plugin == 'fp8':
            ret = self.multiply_gather(quantized_out,
                                       w_quant_out,
                                       gemm_plugin=gemm_plugin,
                                       use_fp8=True,
                                       alpha=alpha,
                                       lora_runtime_params=lora_runtime_params,
                                       lora_hidden_state=lora_hidden_state)
        else:
            dequantized_out = dequantize(quantized_out,
                                         activation_scaling_factor, -1,
                                         self.dtype)
            w_deq_out = dequantize(w_quant_out, weights_scaling_factor, -1,
                                   self.dtype)
            ret = self.multiply_gather(dequantized_out,
                                       w_deq_out,
                                       gemm_plugin=None,
                                       use_fp8=True,
                                       lora_runtime_params=lora_runtime_params,
                                       lora_hidden_state=lora_hidden_state)
        return ret


class FP8RowLinear(RowLinear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size)
        self.weight = Parameter(shape=(self.out_features, self.in_features),
                                dtype='fp8')
        self.activation_scaling_factor = Parameter(shape=(1, ),
                                                   dtype=trt.float32)
        self.weights_scaling_factor = Parameter(shape=(1, ), dtype=trt.float32)

    def forward(self, x, lora_runtime_params=None, reduce_fusion_params=None):
        assert lora_runtime_params is None or default_net(
        ).plugin_config.lora_plugin == self.dtype

        alpha = self.weights_scaling_factor.raw_value * self.activation_scaling_factor.raw_value
        activation_scaling_factor = cast(self.activation_scaling_factor.value,
                                         self.dtype)
        if x.dtype != trt.fp8:
            quantized_out = quantize(x, activation_scaling_factor, 'fp8')
            lora_hidden_state = x if lora_runtime_params is not None else None
        else:
            quantized_out = x
            # TODO: add fp8 LoRA support
            lora_hidden_state = dequantize(
                x, activation_scaling_factor, -1,
                self.dtype) if lora_runtime_params is not None else None

        weights_scaling_factor = cast(self.weights_scaling_factor.value,
                                      self.dtype)
        if self.weight.value.dtype != trt.fp8:
            w_quant_out = quantize(self.weight.value, weights_scaling_factor,
                                   'fp8')
        else:
            w_quant_out = self.weight.value

        gemm_plugin = default_net().plugin_config.gemm_plugin
        if gemm_plugin == 'fp8':
            ret = self.multiply_reduce(
                quantized_out,
                w_quant_out,
                gemm_plugin=gemm_plugin,
                use_fp8=True,
                alpha=alpha,
                lora_runtime_params=lora_runtime_params,
                lora_hidden_state=lora_hidden_state,
                reduce_fusion_params=reduce_fusion_params)
        else:
            dequantized_out = dequantize(quantized_out,
                                         activation_scaling_factor, -1,
                                         self.dtype)
            w_deq_out = dequantize(w_quant_out, weights_scaling_factor, -1,
                                   self.dtype)
            ret = self.multiply_reduce(
                dequantized_out,
                w_deq_out,
                gemm_plugin=None,
                use_fp8=True,
                lora_runtime_params=lora_runtime_params,
                lora_hidden_state=lora_hidden_state,
                reduce_fusion_params=reduce_fusion_params)
        return ret


class Fp8RowwiseMLP(Module):

    def __init__(
            self,
            hidden_size,
            ffn_hidden_size,
            hidden_act,
            bias=True,
            dtype=None,
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
            clamp_val=None,
    ):
        super().__init__()
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        fc_output_size = 2 * ffn_hidden_size if hidden_act == 'swiglu' else ffn_hidden_size
        self.fc = Fp8RowwiseColumnLinear(hidden_size,
                                         fc_output_size,
                                         bias=bias,
                                         dtype=dtype,
                                         tp_group=tp_group,
                                         tp_size=tp_size,
                                         gather_output=False,
                                         quant_mode=quant_mode)

        self.proj = Fp8RowwiseRowLinear(ffn_hidden_size,
                                        hidden_size,
                                        bias=bias,
                                        dtype=dtype,
                                        tp_group=tp_group,
                                        tp_size=tp_size,
                                        quant_mode=quant_mode)

        self.hidden_act = hidden_act
        self.quant_mode = quant_mode
        self.dtype = dtype

        if self.quant_mode.has_act_static_scaling():
            self.quantization_scaling_factor = Parameter(shape=(1, ),
                                                         dtype='float32')
        else:
            self.register_parameter('quantization_scaling_factor', None)

        if clamp_val:
            if not (isinstance(clamp_val, list) and len(clamp_val) == 2):
                raise ValueError(f'unsupported clamp_val {clamp_val}')
            self.clamp_val = Parameter(np.array(clamp_val, dtype=np.float32),
                                       dtype='float32',
                                       is_buffer=True)
        else:
            self.register_parameter('clamp_val', None)

    def forward(self, hidden_states, lora_layer_params=None):
        assert lora_layer_params is None, "lora is not supported on Fp8RowwiseMLP now"
        inter = self.fc(hidden_states)
        inter = ACT2FN[self.hidden_act](inter)
        if self.quant_mode.has_fp8_rowwise():
            # Quantize per token outputs tuple:
            # quantized tensor and scaling factors per token
            inter = quantize_fp8_per_token(inter, self.clamp_val.val)
        output = self.proj(inter)
        return output


class Fp8RowwiseGatedMLP(Fp8RowwiseMLP):

    def __init__(
            self,
            hidden_size,
            ffn_hidden_size,
            hidden_act,
            bias=True,
            dtype=None,
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
            clamp_val=None,
    ):
        super().__init__(hidden_size,
                         ffn_hidden_size,
                         hidden_act,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size,
                         quant_mode=quant_mode,
                         clamp_val=clamp_val)
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        self.gate = Fp8RowwiseColumnLinear(hidden_size,
                                           ffn_hidden_size,
                                           bias=bias,
                                           dtype=dtype,
                                           tp_group=tp_group,
                                           tp_size=tp_size,
                                           gather_output=False,
                                           quant_mode=quant_mode)

        if self.quant_mode.has_act_static_scaling():
            self.quantization_scaling_factor = Parameter(shape=(1, ),
                                                         dtype='float32')
        else:
            self.register_parameter('quantization_scaling_factor', None)

    def forward(self, hidden_states, lora_layer_params=None):
        assert lora_layer_params is None, "lora is not supported on Fp8RowwiseGatedMLP now"
        inter = self.fc(hidden_states)
        inter = ACT2FN[self.hidden_act](inter)
        gate = self.gate(hidden_states)
        inter_x_gate = inter * gate
        if self.quant_mode.has_fp8_rowwise():
            # Quantize per token outputs tuple:
            # quantized tensor and scaling factors per token
            inter_x_gate = quantize_fp8_per_token(inter_x_gate,
                                                  self.clamp_val.value)
        output = self.proj(inter_x_gate)
        return output


class SmoothQuantGatedMLP(SmoothQuantMLP):

    def __init__(
            self,
            hidden_size,
            ffn_hidden_size,
            hidden_act,
            bias=True,
            dtype=None,
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
    ):
        super().__init__(hidden_size,
                         ffn_hidden_size,
                         hidden_act,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size,
                         quant_mode=quant_mode)
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        self.gate = SmoothQuantColumnLinear(hidden_size,
                                            ffn_hidden_size,
                                            bias=bias,
                                            dtype=dtype,
                                            tp_group=tp_group,
                                            tp_size=tp_size,
                                            gather_output=False,
                                            quant_mode=quant_mode)

        if self.quant_mode.has_act_static_scaling():
            self.quantization_scaling_factor = Parameter(shape=(1, ),
                                                         dtype='float32')
        else:
            self.register_parameter('quantization_scaling_factor', None)

    def forward(self, hidden_states, lora_layer_params=None):
        assert lora_layer_params is None, "lora is not supported on SmoothQuantGatedMLP now"
        inter = self.fc(hidden_states)
        inter = ACT2FN[self.hidden_act](inter)
        gate = self.gate(hidden_states)
        inter_x_gate = inter * gate
        smoother = cast(self.proj.smoother.value, self.dtype)
        inter_x_gate = inter_x_gate / smoother
        if self.quant_mode.has_act_and_weight_quant():
            if self.quant_mode.has_act_static_scaling():
                # Avoid quantization layers as it breaks int8 plugins
                inter_x_gate = quantize_tensor(
                    inter_x_gate, self.quantization_scaling_factor.value)
            else:
                # Quantize per token outputs tuple:
                # quantized tensor and scaling factors per token
                inter_x_gate = quantize_per_token(inter_x_gate)

        output = self.proj(inter_x_gate)
        return output


class SmoothQuantAttention(Module):

    def __init__(
            self,
            *,
            local_layer_idx,
            hidden_size,
            num_attention_heads,
            num_kv_heads=None,
            max_position_embeddings=1024,
            num_layers=1,
            apply_query_key_layer_scaling=False,
            attention_head_size=None,
            attention_mask_type=AttentionMaskType.padding,
            bias=True,
            dense_bias=None,
            dtype=None,
            position_embedding_type=PositionEmbeddingType.learned_absolute,
            rotary_embedding_base=10000.0,
            rotary_embedding_scaling=None,
            rotary_embedding_percentage=1.0,
            tp_group=None,
            tp_size=1,
            tp_rank=0,
            scale_alibi_bias=False,
            paged_kv_cache=False,
            quant_mode=QuantMode(0),
    ):
        super().__init__()
        self.local_layer_idx = local_layer_idx
        self.attention_mask_type = attention_mask_type
        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_attention_kv_heads = (
            num_kv_heads + tp_size - 1
        ) // tp_size if num_kv_heads is not None else self.num_attention_heads
        self.hidden_size = hidden_size // tp_size
        self.max_position_embeddings = 0 if max_position_embeddings is None else max_position_embeddings
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dense_bias = dense_bias
        if dense_bias is None:
            self.dense_bias = bias

        self.num_layers = num_layers
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.q_scaling = 1
        if self.apply_query_key_layer_scaling:
            self.norm_factor *= self.num_layers
            self.q_scaling *= self.num_layers
        # Whether to scale ALiBi bias. Mathematically, it's equivalent to
        # normalizing QK after adding bias.
        #   - False, inv_sqrt_Dh * Q*K^T + alibi_bias
        #   - True,  inv_sqrt_Dh * Q*K^T + inv_sqrt_Dh * alibi_bias
        self.scale_alibi_bias = scale_alibi_bias

        self.position_embedding_type = position_embedding_type
        self.paged_kv_cache = paged_kv_cache

        self.rotary_embedding_base = rotary_embedding_base
        self.rotary_embedding_scale_type = RotaryScalingType.none
        self.rotary_embedding_scale = 1.0
        self.rotary_embedding_dim = 0

        if rotary_embedding_scaling is not None:
            self.rotary_embedding_scale_type = RotaryScalingType.from_string(
                rotary_embedding_scaling["type"])
            self.rotary_embedding_scale = rotary_embedding_scaling.get(
                "factor", 1.0)
            assert self.rotary_embedding_scale > 1.0

        if self.position_embedding_type.is_rope():
            self.rotary_embedding_dim = int(self.attention_head_size *
                                            rotary_embedding_percentage)
        elif self.position_embedding_type.is_alibi():
            alibi_scale = 1. / self.norm_factor if self.scale_alibi_bias else 1.
            alibi_slopes = generate_alibi_slopes(self.num_attention_heads *
                                                 self.tp_size,
                                                 tp_size=self.tp_size,
                                                 tp_rank=self.tp_rank,
                                                 alibi_scale=alibi_scale)
            self.register_parameter(
                'alibi_slopes',
                Parameter(alibi_slopes, dtype='float32', is_buffer=True))

        self.quant_mode = quant_mode
        self.dtype = dtype

        if self.quant_mode.has_act_static_scaling():
            self.quantization_scaling_factor = Parameter(shape=(1, ),
                                                         dtype='float32')
        else:
            self.register_parameter('quantization_scaling_factor', None)

        qkv_quant_mode = quant_mode
        if self.quant_mode.has_act_and_weight_quant():
            # We need to hijack quant_mode for QKV because QKV always uses per channel scaling
            qkv_quant_mode = QuantMode.from_description(
                True, True, quant_mode.has_per_token_dynamic_scaling(), True)

        self.register_parameter('kv_cache_scaling_factor', None)

        self.qkv = SmoothQuantColumnLinear(
            hidden_size,
            tp_size * self.num_attention_heads * self.attention_head_size +
            (2 * tp_size * self.num_attention_kv_heads *
             self.attention_head_size),
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            gather_output=False,
            quant_mode=qkv_quant_mode)

        self.dense = SmoothQuantRowLinear(tp_size * self.num_attention_heads *
                                          self.attention_head_size,
                                          hidden_size,
                                          bias=self.dense_bias,
                                          dtype=dtype,
                                          tp_group=tp_group,
                                          tp_size=tp_size,
                                          quant_mode=quant_mode)

        self.use_lora = False

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        use_cache=False,
        kv_cache_params=None,
        attention_params=None,
        spec_decoding_params=None,
        encoder_output=None,
        position_embedding=None,
        norm_before_bmm1=False,
        lora_layer_params=None,
        reduce_fusion_params: Optional[AllReduceFusionParams] = None,
    ):
        assert lora_layer_params is None, "lora is not supported on SmoothQuantAttention now"
        if default_net().plugin_config.smooth_quant_gemm_plugin:
            qkv = self.qkv(hidden_states)
        else:
            raise ValueError("smooth_quant_gemm_plugin is not set")

        alibi_slopes = None
        if self.position_embedding_type == PositionEmbeddingType.alibi:
            alibi_slopes = self.alibi_slopes.value
            dtype = trt.float32
            if default_net().plugin_config.gpt_attention_plugin or default_net(
            ).plugin_config.inflight_batching_gpt_attention_plugin:
                dtype = hidden_states.dtype if self.quant_mode.has_act_static_scaling(
                ) else hidden_states[0].dtype
                if dtype == trt.int8:
                    dtype = trt.float16
            alibi_slopes = cast(alibi_slopes, dtype)

        if spec_decoding_params is None:
            spec_decoding_params = SpecDecodingParams()

        if default_net().plugin_config.gpt_attention_plugin:

            assert attention_params.is_valid(
                default_net().plugin_config.gpt_attention_plugin,
                default_net().plugin_config.remove_input_padding)
            assert kv_cache_params.is_valid(
                default_net().plugin_config.gpt_attention_plugin)
            assert self.attention_mask_type == AttentionMaskType.causal, \
                'Plugin only support masked MHA.'
            if self.kv_cache_scaling_factor is not None:
                kv_orig_quant_scale = constant(fp32_array(
                    [1.0])) / self.kv_cache_scaling_factor.value
                kv_quant_orig_scale = self.kv_cache_scaling_factor.value
            else:
                kv_orig_quant_scale = None
                kv_quant_orig_scale = None
            if self.position_embedding_type.is_rope():
                rotary_inv_freq = attention_params.rotary_inv_freq
                rotary_cos_sin = attention_params.embed_positions_for_gpt_attention
            else:
                rotary_inv_freq = None
                rotary_cos_sin = None
            context, past_key_value = gpt_attention(
                qkv=qkv,
                past_key_value=kv_cache_params.get_first_past_key_value(),
                sequence_length=attention_params.sequence_length,
                host_past_key_value_lengths=kv_cache_params.
                host_past_key_value_lengths,
                host_max_attention_window_sizes=kv_cache_params.
                host_max_attention_window_sizes,
                host_sink_token_length=kv_cache_params.host_sink_token_length,
                context_lengths=attention_params.context_lengths,
                cache_indirection=kv_cache_params.cache_indirection,
                host_request_types=attention_params.host_request_types,
                layer_idx=self.local_layer_idx,
                num_heads=self.num_attention_heads,
                num_kv_heads=self.num_attention_kv_heads,
                hidden_size_per_head=self.attention_head_size,
                q_scaling=self.q_scaling,
                rotary_embedding_dim=self.rotary_embedding_dim,
                rotary_embedding_base=self.rotary_embedding_base,
                rotary_embedding_scale_type=self.rotary_embedding_scale_type,
                rotary_embedding_scale=self.rotary_embedding_scale,
                rotary_embedding_max_positions=self.max_position_embeddings,
                position_embedding_type=self.position_embedding_type,
                rotary_inv_freq=rotary_inv_freq,
                rotary_cos_sin=rotary_cos_sin,
                kv_orig_quant_scale=kv_orig_quant_scale,
                kv_quant_orig_scale=kv_quant_orig_scale,
                kv_cache_quant_mode=self.quant_mode,
                max_context_length=attention_params.max_context_length,
                alibi_slopes=alibi_slopes,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                kv_cache_block_offsets=kv_cache_params.kv_cache_block_offsets,
                host_kv_cache_block_offsets=kv_cache_params.
                host_kv_cache_block_offsets,
                host_kv_cache_pool_pointers=kv_cache_params.
                host_kv_cache_pool_pointers,
                host_context_lengths=attention_params.host_context_lengths,
                spec_decoding_generation_lengths=spec_decoding_params.
                spec_decoding_generation_lengths,
                spec_decoding_position_offsets=spec_decoding_params.
                spec_decoding_position_offsets,
                spec_decoding_packed_mask=spec_decoding_params.
                spec_decoding_packed_mask,
                host_runtime_perf_knobs=attention_params.host_runtime_perf_knobs
            )
        else:
            assert self.paged_kv_cache == False

            def transpose_for_scores(x):
                new_x_shape = concat([
                    shape(x, 0),
                    shape(x, 1), self.num_attention_heads,
                    self.attention_head_size
                ])
                return x.view(new_x_shape).permute([0, 2, 1, 3])

            query, key, value = split(qkv, self.hidden_size, dim=2)
            query = transpose_for_scores(query)
            key = transpose_for_scores(key)
            value = transpose_for_scores(value)

            past_key_value = kv_cache_params.get_first_past_key_value()
            if past_key_value is not None:

                def dequantize_tensor(x, scale):
                    # Cast from int8 to dtype
                    casted_x = cast(x, self.dtype)
                    return casted_x * scale

                if self.quant_mode.has_int8_kv_cache():
                    past_key_value = dequantize_tensor(
                        past_key_value, self.kv_dequantization_scale.value)

                # past_key_value [bs, 2, num_heads, max_seq_len, head_dim]
                past_key, past_value = split(past_key_value, 1, dim=1)

                key_shape = concat([
                    shape(past_key, 0),
                    shape(past_key, 2),
                    shape(past_key, 3),
                    shape(past_key, 4)
                ])
                past_key = past_key.view(key_shape, zero_is_placeholder=False)
                past_value = past_value.view(key_shape,
                                             zero_is_placeholder=False)
                key = concat([past_key, key], dim=2)
                value = concat([past_value, value], dim=2)

            def merge_caches():
                key_inflated_shape = concat([
                    shape(key, 0), 1,
                    shape(key, 1),
                    shape(key, 2),
                    shape(key, 3)
                ])
                inflated_key = key.view(key_inflated_shape,
                                        zero_is_placeholder=False)
                inflated_value = value.view(key_inflated_shape,
                                            zero_is_placeholder=False)
                past_key_value = concat([inflated_key, inflated_value], dim=1)
                return past_key_value

            if self.attention_mask_type == AttentionMaskType.causal:
                query_length = shape(query, 2)
                key_length = shape(key, 2)
                starts = concat([0, 0, key_length - query_length, 0])
                sizes = concat([1, 1, query_length, key_length])
                buffer = constant(
                    np.expand_dims(
                        np.tril(
                            np.ones(
                                (self.max_position_embeddings,
                                 self.max_position_embeddings))).astype(bool),
                        (0, 1)))
                causal_mask = slice(buffer, starts, sizes)

            key = key.permute([0, 1, 3, 2])
            with precision("float32"):
                attention_scores = matmul(query, key)

                if self.attention_mask_type == AttentionMaskType.causal:
                    attention_scores = where(causal_mask, attention_scores,
                                             -10000.0)

                attention_scores = attention_scores / self.norm_factor
                attention_probs = softmax(attention_scores, dim=-1)

            context = matmul(attention_probs, value,
                             use_fp32_acc=False).permute([0, 2, 1, 3])
            context = context.view(
                concat([shape(context, 0),
                        shape(context, 1), self.hidden_size]))

            past_key_value = merge_caches()

            if use_cache and self.quant_mode.has_int8_kv_cache():
                past_key_value = quantize_tensor(
                    past_key_value, self.kv_quantization_scale.value)
        value = cast(self.dense.smoother.value, context.dtype)
        context = context / value
        if self.quant_mode.has_act_and_weight_quant():
            if self.quant_mode.has_act_static_scaling():
                # Avoid quantization layers as it breaks int8 plugins
                context = quantize_tensor(
                    context, self.quantization_scaling_factor.value)
            else:
                # Quantize per token outputs tuple:
                # quantized tensor and scaling factors per token
                context = quantize_per_token(context)

        context = self.dense(
            context,
            reduce_fusion_params=reduce_fusion_params,
        )

        if use_cache:
            return (context, past_key_value)

        return context
