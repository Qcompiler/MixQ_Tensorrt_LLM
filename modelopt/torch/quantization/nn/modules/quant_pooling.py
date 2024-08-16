# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Quantized Pooling modules."""

from torch.nn.modules import pooling

from .quant_module import QuantInputBase, QuantModuleRegistry, _LegacyQuantInputBaseMixin

__all__ = [
    "MaxPool1d",
    "QuantMaxPool1d",
    "MaxPool2d",
    "QuantMaxPool2d",
    "MaxPool3d",
    "QuantMaxPool3d",
    "AvgPool1d",
    "QuantAvgPool1d",
    "AvgPool2d",
    "QuantAvgPool2d",
    "AvgPool3d",
    "QuantAvgPool3d",
    "AdaptiveAvgPool1d",
    "QuantAdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "QuantAdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "QuantAdaptiveAvgPool3d",
]


class QuantMaxPool1d(_LegacyQuantInputBaseMixin, pooling.MaxPool1d):
    """Quantized version of nn.MaxPool1d."""

    pass


class QuantMaxPool2d(_LegacyQuantInputBaseMixin, pooling.MaxPool2d):
    """Quantized version of nn.MaxPool2d."""

    pass


class QuantMaxPool3d(_LegacyQuantInputBaseMixin, pooling.MaxPool3d):
    """Quantized version of nn.MaxPool3d."""

    pass


class QuantAvgPool1d(_LegacyQuantInputBaseMixin, pooling.AvgPool1d):
    """Quantized version of nn.AvgPool1d."""

    pass


class QuantAvgPool2d(_LegacyQuantInputBaseMixin, pooling.AvgPool2d):
    """Quantized version of nn.AvgPool2d."""

    pass


class QuantAvgPool3d(_LegacyQuantInputBaseMixin, pooling.AvgPool3d):
    """Quantized version of nn.AvgPool3d."""

    pass


class QuantAdaptiveAvgPool1d(_LegacyQuantInputBaseMixin, pooling.AdaptiveAvgPool1d):
    """Quantized version of nn.AdaptiveAvgPool1d."""

    pass


class QuantAdaptiveAvgPool2d(_LegacyQuantInputBaseMixin, pooling.AdaptiveAvgPool2d):
    """Quantized version of nn.AdaptiveAvgPool2d."""

    pass


class QuantAdaptiveAvgPool3d(_LegacyQuantInputBaseMixin, pooling.AdaptiveAvgPool3d):
    """Quantized version of nn.AdaptiveAvgPool3d."""

    pass


# Define alias with Quant prefix
MaxPool1d = QuantMaxPool1d
MaxPool2d = QuantMaxPool2d
MaxPool3d = QuantMaxPool3d
AvgPool1d = QuantAvgPool1d
AvgPool2d = QuantAvgPool2d
AvgPool3d = QuantAvgPool3d
AdaptiveAvgPool1d = QuantAdaptiveAvgPool1d
AdaptiveAvgPool2d = QuantAdaptiveAvgPool2d
AdaptiveAvgPool3d = QuantAdaptiveAvgPool3d

QuantModuleRegistry.register({pooling.MaxPool1d: "nn.MaxPool1d"})(QuantInputBase)
QuantModuleRegistry.register({pooling.MaxPool2d: "nn.MaxPool2d"})(QuantInputBase)
QuantModuleRegistry.register({pooling.MaxPool3d: "nn.MaxPool3d"})(QuantInputBase)
QuantModuleRegistry.register({pooling.AvgPool1d: "nn.AvgPool1d"})(QuantInputBase)
QuantModuleRegistry.register({pooling.AvgPool2d: "nn.AvgPool2d"})(QuantInputBase)
QuantModuleRegistry.register({pooling.AvgPool3d: "nn.AvgPool3d"})(QuantInputBase)
QuantModuleRegistry.register({pooling.AdaptiveAvgPool1d: "nn.AdaptiveAvgPool1d"})(QuantInputBase)
QuantModuleRegistry.register({pooling.AdaptiveAvgPool2d: "nn.AdaptiveAvgPool2d"})(QuantInputBase)
QuantModuleRegistry.register({pooling.AdaptiveAvgPool3d: "nn.AdaptiveAvgPool3d"})(QuantInputBase)
