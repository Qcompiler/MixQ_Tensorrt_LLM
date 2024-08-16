# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Hooks for Pytorch to ensure compatibility with dynamic modules."""

from contextlib import ExitStack, nullcontext
from typing import Union

import torch
import torch.nn as nn
from packaging.version import Version

from .dynamic import DynamicModule

if Version(torch.__version__) >= Version("2.1"):
    # Older versions have torch.distributed.fsdp.flat_param module but without `_safe_setattr_tensor_or_param`
    from torch.distributed.fsdp import _flat_param
    from torch.distributed.fsdp._flat_param import FlatParamHandle

    def _safe_setattr_tensor_or_param_with_dm_check(
        module: nn.Module, param_name: str, tensor_or_param: Union[torch.Tensor, nn.Parameter]
    ):
        """A batched version of _safe_setattr_tensor_or_param ensuring compatibility with DMs.

        This function is called during the creation of the flat param in the FSDP module. We intercept
        this call to ensure that dynamic attributes are correctly preserved without interfering with
        FSDP's management, retrieval, and updates of parameters that might also be dynamic attributes.
        """
        with (
            module.reset_dynamic_attributes()
            if isinstance(module, DynamicModule)
            else nullcontext()
        ):
            return _flat_param._safe_setattr_tensor_or_param_original(
                module, param_name, tensor_or_param
            )

    # set the new function as the replacement
    _flat_param._safe_setattr_tensor_or_param_original = _flat_param._safe_setattr_tensor_or_param
    _flat_param._safe_setattr_tensor_or_param = _safe_setattr_tensor_or_param_with_dm_check

    def _writeback_orig_param(self: FlatParamHandle):
        """A batched version of FlatParamHandle._writeback_orig_params ensuring compatibility with DMs.

        This function is called when we use_orig_params=True. We intercept this call to ensure that
        dynamic attributes are correctly preserved without interfering with FSDP's management,
        retrieval, and updates of parameters that might also be dynamic attributes.
        """
        with ExitStack() as stack:
            for _, module, _ in self.flat_param._param_infos:
                if isinstance(module, DynamicModule):
                    stack.enter_context(module.reset_dynamic_attributes())
            return self._writeback_orig_params_original()

    FlatParamHandle._writeback_orig_params_original = FlatParamHandle._writeback_orig_params
    FlatParamHandle._writeback_orig_params = _writeback_orig_param
