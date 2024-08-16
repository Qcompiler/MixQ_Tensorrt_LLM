# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Hardware specific parameters.

All the hardware parameters will be treated as dictionary type for omniengine client APIs.
The key name and the suggested value range are listed with name TENSORRT_HW_PARAMS_SUGGESTED_OPTIONS.
The key name and the optimum value are listed with name TENSORRT_HW_PARAMS_OPT_OPTIONS.
"""
from .constants import (
    DEFAULT_AVG_TIMING,
    DEFAULT_MAX_WORKSPACE_SIZE,
    DEFAULT_MIN_TIMING,
    DEFAULT_TACTIC_SOURCES,
)

# Key names
# Workspace unit: MB
MAX_WORKSPACE_SIZE = [16, 32, 64, 128, 256, 512, 1024]
TACTIC_SOURCES = ["cublasLt", "cublas", "cudnn"]
ALL_TATIC_SOURCES_COMPONENT = []
for source in TACTIC_SOURCES:
    ALL_TATIC_SOURCES_COMPONENT.append("+" + source)
    ALL_TATIC_SOURCES_COMPONENT.append("-" + source)

TENSORRT_HW_PARAMS_SUGGESTED_OPTIONS = {
    "tacticSources": ALL_TATIC_SOURCES_COMPONENT,
    "minTiming": range(1, 5),
    "avgTiming": range(1, 5),
    "workspace": MAX_WORKSPACE_SIZE,
}

TENSORRT_HW_PARAMS_OPT_OPTIONS = {
    "tacticSources": DEFAULT_TACTIC_SOURCES,
    "minTiming": str(DEFAULT_MIN_TIMING),
    "avgTiming": str(DEFAULT_AVG_TIMING),
    "workspace": str(DEFAULT_MAX_WORKSPACE_SIZE),
}
