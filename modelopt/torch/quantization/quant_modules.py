# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Deprecated. Placeholder module for throwing deprecated error."""

from contextlib import contextmanager

from modelopt.torch.utils import DeprecatedError


def initialize(*args, **kwargs):
    """Deprecated. This API is no longer supported.

    Use :meth:`mtq.quantize() <modelopt.torch.quantization.model_quant.quantize>`
    instead to quantize the model.
    """
    raise DeprecatedError(
        "This API is no longer supported. Use `modelopt.torch.quantization.model_quant.quantize()`"
        " API instead to quantize the model."
    )


def deactivate():
    """Deprecated. This API is no longer supported."""
    raise DeprecatedError(
        "This API is no longer supported. Use "
        "`modelopt.torch.quantization.model_quant.disable_quantizer()` API instead "
        "to disable quantization."
    )


@contextmanager
def enable_onnx_export():
    """Deprecated. You no longer need to use this context manager while exporting to ONNX."""
    raise DeprecatedError(
        "You no longer need to use this context manager while exporting to ONNX! please call"
        " `torch.onnx.export` directly."
    )
