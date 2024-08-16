# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Default configurations for sparsity modes."""


from pydantic import create_model

from modelopt.torch.opt.config import ModeloptBaseConfig, get_kwargs_for_create_model_with_rules

from .module import SpDMRegistry

SparseMagnitudeConfig = create_model(
    "SparseMagnitudeConfig",
    **get_kwargs_for_create_model_with_rules(
        registry=SpDMRegistry,
        default_rules={
            "nn.Linear": {"*": {}, "*lm_head*": None},
            "nn.Conv2d": {"*": {}, "*lm_head*": None},
        },
        doc='Configuration for the ``"sparse_magnitude"`` mode.',
    ),
)


SparseGPTConfig = create_model(
    "SparseGPTConfig",
    **get_kwargs_for_create_model_with_rules(
        registry=SpDMRegistry,
        default_rules={
            "nn.Linear": {"*": {}, "*lm_head*": None},
            "nn.Conv2d": {"*": {}, "*lm_head*": None},
        },
        doc='Configuration for the ``"sparse_gpt"`` mode.',
    ),
)


class ExportSparseConfig(ModeloptBaseConfig):
    """Configuration (empty!) for the ``"export_sparse"`` mode."""
