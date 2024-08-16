# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Handles sparsity plugins for third-party modules.

Currently, we support plugins for

- :meth:`megatron<modelopt.torch.sparsity.plugins.megatron>`

"""

import warnings

try:
    from .megatron import *

    has_megatron_core = True
except ImportError:
    has_megatron_core = False
except Exception as e:
    has_megatron_core = False
    warnings.warn(f"Failed to import megatron plugin due to: {repr(e)}")
