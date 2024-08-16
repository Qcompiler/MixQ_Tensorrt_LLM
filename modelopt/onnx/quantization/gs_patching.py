# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Patch onnx_graphsurgeon to support explicitly setting a dtype."""

from typing import Sequence, Union

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx.mapping import TENSOR_TYPE_MAP
from onnx_graphsurgeon.ir.tensor import LazyValues

from modelopt.onnx.quantization.quant_utils import pack_float32_to_4bit_optimized

GS_LOGGER = gs.logger.logger.G_LOGGER
ONNX_MAJOR, ONNX_MINOR = tuple(map(int, onnx.__version__.split(".")[:2]))


def _onnx_supports_int4():
    return ONNX_MAJOR > 1 or ONNX_MAJOR == 1 and ONNX_MINOR >= 16


def _make_constant(
    name: str, values: Union[np.ndarray, LazyValues], dtype: onnx.TensorProto.DataType
) -> gs.Constant:
    """Creates a constant with a specified dtype."""
    converted_dtype = (
        dtype if isinstance(values, LazyValues) else TENSOR_TYPE_MAP[int(dtype)].np_dtype
    )
    if values.dtype != converted_dtype:
        GS_LOGGER.critical(
            f"Trying to create tensor with incompatible types: `{values.dtype}`, `{dtype}`"
        )

    t = gs.Constant(name, values)
    setattr(t, "explicit_dtype", dtype)
    return t


def _make_variable(
    name: str, dtype: onnx.TensorProto.DataType, shape: Sequence[Union[int, str]]
) -> gs.Constant:
    """Creates a variable with a specified dtype."""
    x = gs.Variable(name, TENSOR_TYPE_MAP[int(dtype)].np_dtype, shape)
    setattr(x, "explicit_dtype", dtype)
    return x


def _export_tensor_proto(tensor: gs.Constant) -> onnx.TensorProto:
    if isinstance(tensor._values, LazyValues):
        onnx_tensor = tensor._values.tensor
    else:
        # is numpy array.
        dtype = getattr(
            tensor, "explicit_dtype", onnx.helper.np_dtype_to_tensor_dtype(tensor.values.dtype)
        )

        vals = tensor.values
        if _onnx_supports_int4() and dtype in [onnx.TensorProto.INT4, onnx.TensorProto.UINT4]:
            signed = dtype == onnx.TensorProto.INT4
            np_dtype = onnx.helper.tensor_dtype_to_np_dtype(dtype)
            vals = pack_float32_to_4bit_optimized(tensor.values, signed=signed).astype(np_dtype)

        onnx_tensor = onnx.helper.make_tensor(
            tensor.name,
            dtype,
            dims=tensor.values.shape,
            vals=vals.tobytes(),
            raw=True,
        )
        if tensor.data_location is not None:
            onnx_tensor.data_location = tensor.data_location
    onnx_tensor.name = tensor.name
    return onnx_tensor


def _export_value_info_proto(tensor: gs.Variable, do_type_check: bool) -> onnx.ValueInfoProto:
    if do_type_check and tensor.dtype is None:
        GS_LOGGER.critical(
            "Graph input and output tensors must include dtype information. Please set the dtype"
            " attribute for: {:}".format(tensor)
        )

    if tensor.dtype is not None:
        dtype = getattr(
            tensor, "explicit_dtype", onnx.helper.np_dtype_to_tensor_dtype(np.dtype(tensor.dtype))
        )
        onnx_tensor = onnx.helper.make_tensor_value_info(tensor.name, dtype, tensor.shape)
    else:
        onnx_tensor = onnx.helper.make_empty_tensor_value_info(tensor.name)
    return onnx_tensor


def patch_gs_modules():
    """Dynamically patch graphsurgeon modules."""
    gs.make_constant = _make_constant
    gs.make_variable = _make_variable
    gs.exporters.onnx_exporter.OnnxExporter.export_tensor_proto = _export_tensor_proto
    gs.exporters.onnx_exporter.OnnxExporter.export_value_info_proto = _export_value_info_proto
