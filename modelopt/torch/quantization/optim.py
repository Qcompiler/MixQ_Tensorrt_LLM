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

from modelopt.torch.utils import DeprecatedError


def match_parameters(*args, **kwargs):
    """Deprecated. Placeholder function for throwing deprecated error."""
    raise DeprecatedError("This method is deprecated. ")


def group_parameters(*args, **kwargs):
    """Deprecated. Placeholder function for throwing deprecated error."""
    raise DeprecatedError("This method is deprecated. ")


def freeze_parameters(*args, **kwargs):
    """Deprecated. Placeholder function for throwing deprecated error."""
    raise DeprecatedError("This module is deprecated. ")


def quant_weight_inplace(*args, **kwargs):
    """Deprecated. Placeholder function for throwing deprecated error."""
    raise DeprecatedError("This method is deprecated. ")
