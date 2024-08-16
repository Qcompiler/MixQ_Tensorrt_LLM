# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from .registry import *
from .runtime_client import *

# no runtime_client_impl will be available if 'deploy' is not installed
try:
    from .ort_client import *
except ImportError:
    pass

try:
    from .trt_client import *
except (ImportError, ModuleNotFoundError):
    # ImportError if tensorrt is not installed
    # ModuleNotFoundError if .tensorrt/ is not available
    pass
