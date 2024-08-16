# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Handles plugins for third-party modules."""

import warnings

try:
    from .mcore_dist_checkpointing import *
except ImportError:
    pass
except Exception as e:
    warnings.warn(f"Failed to import megatron core dist checkpointing plugin due to: {repr(e)}")
