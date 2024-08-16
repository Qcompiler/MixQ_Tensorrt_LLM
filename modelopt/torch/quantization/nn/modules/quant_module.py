# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Base class for quantization modules."""
import contextlib
from typing import Union

import torch

from modelopt.torch.opt.dynamic import DynamicModule, _DMRegistryCls
from modelopt.torch.quantization.utils import is_torch_export_mode

from ...tensor_quant import QUANT_DESC_8BIT_PER_TENSOR
from .tensor_quantizer import SequentialQuantizer, TensorQuantizer

__all__ = ["QuantInputBase", "QuantLinearConvBase", "QuantModuleRegistry"]

QuantModuleRegistry = _DMRegistryCls("Quant")


class QuantInputBase(DynamicModule):
    """Base class for modules where the input is quantized."""

    input_quantizer: Union[TensorQuantizer, SequentialQuantizer]
    output_quantizer: Union[TensorQuantizer, SequentialQuantizer]
    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_output = QUANT_DESC_8BIT_PER_TENSOR

    def forward(self, input, *args, **kwargs):
        """Quantize the input before calling the original forward method."""
        input = self.input_quantizer(input)
        output = super().forward(input, *args, **kwargs)
        if isinstance(output, tuple):
            return (self.output_quantizer(output[0]), *output[1:])
        return self.output_quantizer(output)

    def _setup(self):
        """Patch the module's forward method to quantize the input."""
        self._register_temp_attribute(
            "input_quantizer", TensorQuantizer(self.default_quant_desc_input)
        )
        self._register_temp_attribute(
            "output_quantizer", TensorQuantizer(self.default_quant_desc_output)
        )
        self.output_quantizer.disable()


class QuantLinearConvBase(QuantInputBase):
    """Base class for quantized linear modules.

    Quantized linear modules are modules where both the input and the weight are quantized.
    """

    weight_quantizer: Union[TensorQuantizer, SequentialQuantizer]
    _enable_weight_quantization: bool
    default_quant_desc_weight = QUANT_DESC_8BIT_PER_TENSOR

    @contextlib.contextmanager
    def quantize_weight(self):
        """Context in which `self.weight` is quantized."""
        self._enable_weight_quantization = True
        yield
        self._enable_weight_quantization = False

    @staticmethod
    def _get_quantized_weight(module: "QuantLinearConvBase", weight: torch.Tensor) -> torch.Tensor:
        if module._enable_weight_quantization or is_torch_export_mode():
            return module.weight_quantizer(weight)
        return weight

    def forward(self, input, *args, **kwargs):
        """Quantize the input and the weight before calling the original forward method."""
        # self.quntize_weight() setting attributes is not allowed for torch.export.
        if is_torch_export_mode():
            return super().forward(input, *args, **kwargs)
        with self.quantize_weight():
            return super().forward(input, *args, **kwargs)

    def _setup(self):
        super()._setup()
        self._register_temp_attribute(
            "weight_quantizer", TensorQuantizer(self.default_quant_desc_weight)
        )
        self._register_temp_attribute("_enable_weight_quantization", False)
        self._register_dynamic_attribute("weight", self._get_quantized_weight)

    @staticmethod
    def initialize_quantizer_with_dummy_states(module):
        """Initialize the quantizer states with dummy values with the correct type and device."""
        # Lets do a local import; nn modules should not import from model_calib
        from modelopt.torch.quantization.model_calib import max_calibrate

        def _initialize_activation_quantizer_amax(quantizer, device, dtype):
            if not getattr(quantizer, "_has_amax", False):
                return
            # We need the outputs to initialize the amax in this case; Weights alone are not enough
            if (
                quantizer.block_sizes is not None
                and quantizer.block_sizes.get("type", None) != "dynamic"
            ):
                return
            if quantizer.axis is not None:
                return
            quantizer.amax = torch.tensor(1, device=device, dtype=dtype)

        device, dtype = module.weight.device, module.weight.dtype

        for input_quantizer in SequentialQuantizer.tensor_quantizer_iterator(
            getattr(module, "input_quantizer", None)
        ):
            if getattr(input_quantizer, "_has_pre_quant_scale", False):
                input_quantizer.pre_quant_scale = torch.ones(
                    module.weight.shape[1], device=device, dtype=dtype
                )

            _initialize_activation_quantizer_amax(input_quantizer, device, dtype)

        for output_quantizer in SequentialQuantizer.tensor_quantizer_iterator(
            getattr(module, "output_quantizer", None)
        ):
            _initialize_activation_quantizer_amax(output_quantizer, device, dtype)

        for weight_quantizer in SequentialQuantizer.tensor_quantizer_iterator(
            getattr(module, "weight_quantizer", None)
        ):
            if getattr(weight_quantizer, "_has_amax", False):
                max_calibrate(
                    weight_quantizer, lambda weight_quantizer: weight_quantizer(module.weight)
                )


class _LegacyQuantInputBaseMixin:
    """A mixin to support legacy quantized modules which needs to have an __init__ method."""

    _quantized_cls = QuantInputBase
    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_output = QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, *args, quant_desc_input=None, **kwargs):
        """Initialize the module with its original __init__ and patch its forward."""
        self.default_quant_desc_input = quant_desc_input or self.default_quant_desc_input
        super().__init__(*args, **kwargs)
        QuantModuleRegistry.convert(self)


class _LegacyQuantLinearConvBaseMixin(_LegacyQuantInputBaseMixin):
    """A mixin to support legacy quantized modules which needs to have an __init__ method."""

    _quantized_cls = QuantLinearConvBase
    default_quant_desc_weight = QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, *args, quant_desc_input=None, quant_desc_weight=None, **kwargs):
        """Initialize the module with its original __init__ and patch its forward."""
        self.default_quant_desc_weight = quant_desc_weight or self.default_quant_desc_weight
        super().__init__(*args, quant_desc_input=quant_desc_input, **kwargs)
