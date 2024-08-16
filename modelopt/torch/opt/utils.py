# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utilities for optimization."""

from typing import Generator, Optional, Tuple

import torch.nn as nn

from modelopt.torch.utils import unwrap_model

from .dynamic import DynamicSpace
from .hparam import Hparam


class _DynamicSpaceUnwrapped(DynamicSpace):
    """A wrapper for the DynamicSpace class to handle unwrapping of model wrappers like DDP.

    This is useful to ensure that configurations are valid for both vanilla models and wrapped
    models, see :meth:`unwrap_models<modelopt.torch.utils.network.unwrap_model>` for supported wrappers.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__(unwrap_model(model))


def is_configurable(model: nn.Module) -> bool:
    """Check if the model is configurable."""
    return _DynamicSpaceUnwrapped(model).is_configurable()


def is_dynamic(model: nn.Module) -> bool:
    """Check if the model is dynamic."""
    return _DynamicSpaceUnwrapped(model).is_dynamic()


def named_hparams(
    model, configurable: Optional[bool] = None
) -> Generator[Tuple[str, Hparam], None, None]:
    """Recursively yield the name and instance of *all* hparams."""
    yield from _DynamicSpaceUnwrapped(model).named_hparams(configurable)


def get_hparam(model, name: str) -> Hparam:
    """Get the hparam with the given name."""
    return _DynamicSpaceUnwrapped(model).get_hparam(name)


def search_space_size(model: nn.Module) -> int:
    """Return the size of the search space."""
    return _DynamicSpaceUnwrapped(model).size()
