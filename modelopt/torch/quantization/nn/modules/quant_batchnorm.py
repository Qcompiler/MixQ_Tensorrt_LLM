# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Quantized batch normalization module."""

import torch.nn as nn

from .quant_module import QuantInputBase, QuantModuleRegistry

QuantModuleRegistry.register({nn.BatchNorm1d: "nn.BatchNorm1d"})(QuantInputBase)
QuantModuleRegistry.register({nn.BatchNorm2d: "nn.BatchNorm2d"})(QuantInputBase)
QuantModuleRegistry.register({nn.BatchNorm3d: "nn.BatchNorm3d"})(QuantInputBase)
