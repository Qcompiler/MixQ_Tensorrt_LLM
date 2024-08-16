# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Base Class for Real Quantized Tensor."""

import enum

import torch


class QTensorType(enum.Enum):
    """Enumeration for defining types of quantization."""

    INT4 = 1
    INT8 = 2
    FP8 = 3
    NF4 = 4


__all__ = ["BaseQuantizedTensor", "QTensorWrapper"]


class BaseQuantizedTensor:
    """Base class for quantized tensors, providing methods for quantization and dequantization.

    This class should be subclassed to implement specific types of quantized tensors. It handles the
    storage of quantized data along with the necessary configurations and original attributes.

    Attributes:
        original_meta_tensor (torch.Tensor): Original meta to keep attributes of original tensors.
        quantized_data (torch.Tensor): Storage for the quantized tensor data. Quantized_data dtype is
            customized per QuantizedTensor implementation.
    """

    _original_meta_tensor: torch.Tensor
    _quantized_data: torch.Tensor

    def __init__(
        self,
        original_meta_tensor: torch.Tensor,
        quantized_data: torch.Tensor,
    ):
        """Initialize data attributes."""
        self._original_meta_tensor = original_meta_tensor
        self._quantized_data = quantized_data

    @classmethod
    def quantize(cls, input: torch.Tensor, block_size: int):
        """Pack a fake torch.Tensor into a real quantized tensor.

        Args:
            fake_quant_tensor (torch.Tensor): The fake quantized tensor.

        Returns:
            A real quantized tensor, scales.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def dequantize(self, dtype: torch.dtype, **kwarg):
        """Converts the quantized tensor back to a standard torch.Tensor.

        Returns:
            torch.Tensor: The dequantized tensor.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")


class QTensorWrapper(torch.nn.Parameter):
    """A wrapper class for quantized tensors to make them compatible with torch.nn.Parameter.

    Args:
        qtensor (BaseQuantizedTensor): The quantized tensor to be wrapped.
    """

    def __new__(cls, qtensor: BaseQuantizedTensor):
        """Create a new QTensorWrapper instance."""
        quantized_tensor = qtensor._quantized_data.view(torch.int8)
        instance = super().__new__(cls, quantized_tensor, requires_grad=False)
        instance._qtensor = qtensor
        return instance
