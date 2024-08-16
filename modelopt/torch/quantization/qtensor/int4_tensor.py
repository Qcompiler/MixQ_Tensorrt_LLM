# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Implements INT4 quantization for efficient tensor storage and computation."""

import torch

from modelopt.torch.quantization.extensions import get_cuda_ext
from modelopt.torch.quantization.qtensor.base_qtensor import BaseQuantizedTensor
from modelopt.torch.quantization.utils import reduce_amax

__all__ = ["INT4QTensor"]


class INT4QTensor(BaseQuantizedTensor):
    """Implements the INT4 quantization on tensors for more efficient storage or computation.

    Attributes:
        quantized_data (torch.Tensor): The quantized data stored as a packed uint8 tensor.
    """

    @staticmethod
    def _get_quant_maxbound(num_bits):
        return 2 ** (num_bits - 1) - 1

    @classmethod
    def quantize(cls, input: torch.Tensor, block_size: int) -> torch.Tensor:
        """Converting a tensor to a quantized format based on INT4 (AWQ) quantization.

        Args:
            input (torch.Tensor): The input tensor to be quantized.
            block_size (int): The size of each block for quantization.

        Returns:
            tuple: Contains quantized data, input quantization config, and scale quantization config.
        """
        cuda_ext = get_cuda_ext()

        scale_quant_maxbound = cls._get_quant_maxbound(num_bits=4)

        # get scales for each block
        assert (
            input.numel() % block_size == 0
        ), "Number of input elements is not divisible by the block size."
        block_input = input.view(-1, block_size)
        scales = scale_quant_maxbound / reduce_amax(block_input, -1)
        # expand scalers to match shape of input
        scales = scales.view(block_input.shape[0], -1)  # shape: (block_input.shape[0], 1)

        if cuda_ext and input.is_cuda:
            # use a custom cuda kernel if available
            packed_output_uint8 = cuda_ext.INT4_quantize(input, scales, block_size)
        else:
            scaled_blocks = block_input * scales
            flattened = scaled_blocks.flatten()
            # uint4: 0 - 15
            flattened = flattened.round().clamp(
                -(scale_quant_maxbound + 1), scale_quant_maxbound
            ) + (scale_quant_maxbound + 1)
            flattened = flattened.to(torch.uint8)

            packed_output_uint8 = torch.empty(
                input.numel() // 2, dtype=torch.uint8, device=input.device
            )
            # pack the int4 weights into a uint8 tensor
            # packing format: w0, w1, w2, w3, w4, w5, ...
            #               | byte  | byte  | byte  |
            packed_output_uint8 = flattened[::2] << 4 | flattened[1::2]

        return cls(input.to(device="meta"), packed_output_uint8), scales

    def dequantize(self, dtype: torch.dtype, **kwarg):
        """Dequantze INT4 packed tensor to a target dtype."""
        cuda_ext = get_cuda_ext()

        # get kwargs
        scales = kwarg["scale"]
        block_sizes = kwarg["block_sizes"]
        scale_quant_maxbound = self._get_quant_maxbound(num_bits=4)

        if cuda_ext and self._quantized_data.is_cuda:
            # use a custom cuda kernel if available
            output = cuda_ext.INT4_dequantize(self._quantized_data, scales, block_sizes[-1])
            return output.view(self._original_meta_tensor.shape).to(dtype)
        else:
            # indexing in torch required long dtype, we may need to optimize this with customized kernels
            # convert (0, 15) -> (-8, 7)
            first_half = (self._quantized_data >> 4).to(torch.long) - (scale_quant_maxbound + 1)
            second_half = (self._quantized_data & 0x0F).to(torch.long) - (scale_quant_maxbound + 1)

            # de-quantize tensor
            first_half = first_half.view(-1, block_sizes[-1] // 2) / scales.view(-1, 1)
            second_half = second_half.view(-1, block_sizes[-1] // 2) / scales.view(-1, 1)

            # merge the interleaving elements
            first_half = first_half.flatten().unsqueeze(-1).transpose(0, 1)
            second_half = second_half.flatten().unsqueeze(-1).transpose(0, 1)
            return (
                torch.stack([first_half, second_half], dim=-1)
                .reshape(self._original_meta_tensor.shape)
                .to(dtype)
            )
