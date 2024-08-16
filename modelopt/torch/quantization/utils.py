# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Quantization utilities."""

from contextlib import ExitStack, contextmanager

import torch

__all__ = [
    "reduce_amax",
    "is_quantized",
    "is_quantized_layer_with_weight",
    "is_quantized_column_parallel_linear",
    "is_quantized_row_parallel_linear",
    "replace_function",
    "EXPORT_MODE",
    "export_torch_mode",
    "is_torch_library_supported",
]


def reduce_amax(input, axis=None, keepdims=True):
    """Compute the absolute maximum value of a tensor.

    Reduces input_tensor along the dimensions given in axis. Unless keepdims is true,
    the rank of the tensor is reduced by 1 for each entry in axis. If keepdims is true,
    the reduced dimensions are retained with length 1.

    .. note::
        Gradient computation is disabled as this function is never meant learning reduces amax

    Args:
        input: Input tensor
        axis: The dimensions to reduce. None or int or tuple of ints. If None (the default),
            reduces all dimensions. Must be in the range [-rank(input_tensor), rank(input_tensor)).
        keepdims: A boolean. If true, retains reduced dimensions with length 1. Default True
        granularity: DEPRECTED. specifies if the statistic has to be calculated at tensor or channel granularity

    Returns:
        The reduced tensor.

    Raises:
        ValueError: Any axis which doesn't make sense or is not supported
        ValueError: If unknown granularity is passed in.
    """
    with torch.no_grad():
        # A memory-efficient implementation that avoids copying input tensor
        if axis is None:
            max_val = torch.max(input)
            min_val = torch.min(input)
            output = torch.maximum(torch.abs(max_val), torch.abs(min_val))
        else:
            if isinstance(axis, int):
                axis = (axis,)
            max_val = torch.amax(input, dim=axis, keepdim=keepdims)
            min_val = torch.amin(input, dim=axis, keepdim=keepdims)
            output = torch.maximum(torch.abs(max_val), torch.abs(min_val))
            if output.numel() == 1:
                output.squeeze_()
        return output


def is_quantized(module):
    """Check if a module is quantized."""
    from modelopt.torch.quantization.nn import TensorQuantizer

    for _module in module.modules():
        if isinstance(_module, TensorQuantizer):
            return True
    return False


def is_quantized_layer_with_weight(module):
    """Check if a module is quantized with weights."""
    return is_quantized(module) and getattr(module, "weight", None) is not None


def is_quantized_linear(module):
    """Check if a module is a quantized linear module."""
    return (
        hasattr(module, "input_quantizer")
        and hasattr(module, "weight_quantizer")
        and getattr(module, "weight", None) is not None
        and module.weight.dim() == 2
    )


def _is_quantized_parallel_linear(module, mod_name):
    parallel_layers = []
    try:
        import apex.transformer.tensor_parallel.layers as apex_parallel

        parallel_layers.append(getattr(apex_parallel, mod_name))
    except ImportError:
        pass

    try:
        import megatron.core.tensor_parallel.layers as megatron_parallel

        parallel_layers.append(getattr(megatron_parallel, mod_name))
    except ImportError:
        pass

    return is_quantized_linear(module) and isinstance(module, tuple(parallel_layers))


def is_quantized_column_parallel_linear(module):
    """Check if a module is a quantized column parallel linear module."""
    return _is_quantized_parallel_linear(module, "ColumnParallelLinear")


def is_quantized_row_parallel_linear(module):
    """Check if a module is a quantized row parallel linear module."""
    return _is_quantized_parallel_linear(module, "RowParallelLinear")


@contextmanager
def replace_function(package, name, new_func):
    """Replace a function with a new one within a context."""
    old_func = getattr(package, name)
    setattr(package, name, new_func)
    setattr(package, "_" + name, old_func)
    yield
    setattr(package, name, old_func)
    delattr(package, "_" + name)


@contextmanager
def multi_context(*cms):
    """Context manager enabling variable number of context managers."""
    with ExitStack() as stack:
        yield [stack.enter_context(cls) for cls in cms]


EXPORT_MODE: bool = False


@contextmanager
def export_torch_mode():
    """Context manager enabling the export mode."""
    global EXPORT_MODE
    original_value = EXPORT_MODE
    EXPORT_MODE = True
    try:
        yield
    finally:
        EXPORT_MODE = original_value


def is_torch_export_mode():
    """Check whether in the context of exporting model to torch."""
    return EXPORT_MODE


def is_torch_library_supported():
    """Check if the installed PyTorch version meets or exceeds a specified version."""
    ver_strs = torch.__version__.split(".")
    major, minor = ver_strs[0], ver_strs[1]
    # Require torch version >= 2.2.0
    # Adding checks for `impl` and `impl_abstract` as they are experiemental features
    return (
        major >= "2"
        and minor >= "2"
        and hasattr(torch.library, "impl")
        and hasattr(torch.library, "impl_abstract")
    )
