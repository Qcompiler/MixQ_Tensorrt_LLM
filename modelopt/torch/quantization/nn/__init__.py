# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Modules with quantization support."""

import torch
from packaging.version import Version

if Version(torch.__version__) >= Version("2.0"):
    from .modules.quant_rnn import *

from .modules.clip import *
from .modules.quant_activations import *
from .modules.quant_batchnorm import *
from .modules.quant_conv import *
from .modules.quant_instancenorm import *
from .modules.quant_linear import *
from .modules.quant_module import *
from .modules.quant_pooling import *
from .modules.tensor_quantizer import *
