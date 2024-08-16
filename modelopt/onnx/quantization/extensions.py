# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Module to load C++ extensions."""

import os
import sys
import warnings

import cppimport

try:
    print("Loading extension modelopt_round_and_pack_ext...\n")
    path = os.path.join(os.path.dirname(__file__), "src")
    sys.path.append(path)
    round_and_pack_ext = cppimport.imp("modelopt_round_and_pack_ext")
    sys.path.remove(path)
except Exception as e:
    warnings.warn(
        f"{e}\nUnable to load `modelopt_round_and_pack_ext', falling back to python based optimized version."
    )
    round_and_pack_ext = None
