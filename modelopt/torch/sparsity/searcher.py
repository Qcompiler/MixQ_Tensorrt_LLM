# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Searcher interface for sparsity algorithms."""

from abc import abstractmethod
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn

from modelopt.torch.opt.searcher import BaseSearcher, SearchConfig, SearchStateDict
from modelopt.torch.utils import print_rank_0

from . import magnitude as asp
from .module import SparseModule


class BaseSparseSearcher(BaseSearcher):
    """A generic sparse mask searching algorithm."""

    _pattern_2_4 = "2:4 sparsity"

    @property
    def default_search_config(self) -> SearchConfig:
        """Get the default config for the searcher."""
        return {
            **super().default_search_config,
            "score_func": lambda _: 0.0,
            "pattern": self._pattern_2_4,
        }

    @property
    def default_state_dict(self) -> SearchStateDict:
        """Return default state dict."""
        return {}

    def sanitize_search_config(self, config: Optional[SearchConfig]) -> SearchStateDict:
        """Sanitize the search config dict."""
        config_sanitized = super().sanitize_search_config(config)

        # sanity check of sparsity format
        is_nm_prune, n, m = asp.get_nmprune_info(config_sanitized["pattern"])
        assert (
            is_nm_prune and n == 2 and m == 4
        ), f"Unsupported pattern {self.config['pattern']} for sparsity"

        return config_sanitized

    @abstractmethod
    def _check_weight_size(self, weight, mod_name) -> bool:
        """Check if the weight size is supported by the algorithm."""
        raise NotImplementedError

    @abstractmethod
    def _compute_mask(self, module: SparseModule) -> torch.BoolTensor:
        """Compute the mask and update weight for a given module."""
        raise NotImplementedError

    def _named_sparsifiable_modules(self) -> Iterator[Tuple[str, nn.Module]]:
        """Get the named sparsifiable modules."""
        for name, module in self.model.named_modules():
            if (
                isinstance(module, SparseModule)
                and module.is_sparse
                and self._check_weight_size(module.weight, name)
            ):
                yield name, module

    def run_search(self):
        """Search for sparse mask."""
        for name, module in self._named_sparsifiable_modules():
            # compute the mask (and potentially weight update inside compute_mask)
            print_rank_0(f"Searching for sparse mask and weight update for module {name}.")
            with torch.no_grad():
                module.set_mask(self._compute_mask(module))
