# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Provides some basic utilities that can be used in quantize() methods."""

from typing import Sequence, Union

import numpy as np

from .extensions import round_and_pack_ext

INT4_MIN = -8
INT4_MAX = 7
UINT4_MIN = 0
UINT4_MAX = 15


def pack_float32_to_4bit_optimized(array: Union[np.ndarray, Sequence], signed: bool) -> np.ndarray:
    """Convert an array of float32 value to a 4bit data-type and pack every two concecutive elements in a byte.

    This is the optimized version of pack_float32_to_4bit() utility in ONNX helper file. The basic optimizations
    done here mainly rely on moving some common code out of the per-element function calls or loops, thereby making
    them per-input-array, instead of per-input-element. The remaining logic should largely remain as is.

    Args:
        array: array of float to convert and pack
        signed: Whether the 4 bit variant is signed or unsigned

    Returns:
        Packed array with size `ceil(array.size/2)` (single dimension).
    """
    if not isinstance(array, np.ndarray):
        array = np.asarray(array, dtype=np.float32)

    array_flat = array.ravel()
    is_odd_volume = np.prod(array.shape) % 2 == 1
    if is_odd_volume:
        array_flat = np.append(array_flat, np.array([0]))

    inp_arr_len = array_flat.size
    dtype = np.int8 if signed else np.uint8
    clip_low = INT4_MIN if signed else UINT4_MIN
    clip_high = INT4_MAX if signed else UINT4_MAX
    array_flat = np.clip(array_flat, clip_low, clip_high)
    array_flat = np.rint(array_flat).astype(dtype)
    assert len(array_flat) % 2 == 0, "array length must be even at this point"
    assert len(array_flat) == inp_arr_len, "output-length must match the input-length"
    output_list = []
    for i in range(0, inp_arr_len, 2):
        output_list.append((array_flat[i + 1] << 4) | (array_flat[i] & 0x0F))
    arr = np.array(output_list)
    return arr.astype(np.uint8)


def pack_float32_to_4bit_cpp_based(array: Union[np.ndarray, Sequence], signed: bool) -> np.ndarray:
    """Convert an array of float32 value to a 4bit data-type and pack every two concecutive elements in a byte.

    This is the optimized version of pack_float32_to_4bit() utility in ONNX helper file. The basic optimizations
    here is to implement this round_and_pack logic in C++, which is supposed to be faster.

    Args:
        array: array of float to convert and pack
        signed: Whether the 4 bit variant is signed or unsigned

    Returns:
        Packed array with size `ceil(array.size/2)` (single dimension).
    """
    if not isinstance(array, np.ndarray):
        array = np.asarray(array, dtype=np.float32)

    # - Currently, FP32, FP64, UINT8 and INT8 dtypes have C++ implementation of the round-and-pack
    # logic.
    # - With above, FP16 should also get "implicitly" supported due to possible type promotion to higher
    # precision float types (e.g. float32 or float64). So, it is mentioned as supported below.
    # - We can add support for other dtypes as and when needed.
    use_python_version = False
    if round_and_pack_ext is None or array.dtype not in [
        "float",
        "float16",
        "float32",
        "float64",
        "int8",
        "uint8",
    ]:
        use_python_version = True

    array_flat = array.ravel()
    is_odd_volume = np.prod(array.shape) % 2 == 1
    if is_odd_volume:
        array_flat = np.append(array_flat, np.array([0], array_flat.dtype))

    inp_arr_len = array_flat.size

    assert inp_arr_len % 2 == 0, "input array length must be even at this point"

    if use_python_version:
        print(
            f"Using python optimized version for round_and_pack...input-array-dtype={array_flat.dtype}\n"
        )
        numpy_out = pack_float32_to_4bit_optimized(array_flat, signed)
    else:
        numpy_out = np.zeros([1, int(inp_arr_len / 2)], dtype=np.int8)
        numpy_out = numpy_out.ravel()
        ret = round_and_pack_ext.round_and_pack(
            signed, array_flat, array_flat.size, numpy_out, numpy_out.size
        )
        assert ret == inp_arr_len / 2, "Unexpected output length"
        numpy_out = numpy_out.astype(np.uint8)

    return numpy_out
