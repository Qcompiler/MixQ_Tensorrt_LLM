# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Quantization conversion/restore utilities."""

import fnmatch
from typing import Any, Callable, Dict, List, Union

import torch.nn as nn

from modelopt.torch.opt.conversion import ApplyModeError, ModelLikeModule
from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict

from .config import (
    QuantizeConfig,
    QuantizeQuantCfgType,
    QuantizerAttributeConfig,
    _QuantizeExportConfig,
)
from .nn import QuantLinearConvBase, QuantModuleRegistry, SequentialQuantizer, TensorQuantizer
from .plugins.custom import register_custom_model_plugins_on_the_fly
from .utils import is_quantized, is_quantized_layer_with_weight

__all__ = [
    "replace_quant_module",
    "set_quantizer_by_cfg",
    "set_quantizer_attribute",
    "register",
    "unregister",
]


def convert_to_quantized_model(model: nn.Module, config: QuantizeConfig) -> ConvertReturnType:
    """Convert the model to a quantized one as per `config`."""
    # initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    replace_quant_module(model)
    set_quantizer_by_cfg(model, config["quant_cfg"])

    metadata = {"quantizer_state": quantizer_state(model)}

    return model, metadata


def restore_quantized_model(
    model: nn.Module, config: QuantizeConfig, metadata: MetadataDict
) -> nn.Module:
    """Restores the quantizer states from the given state dict."""
    # initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    assert not is_quantized(model), "Model must not be quantized!"

    quantizer_state_dict = metadata["quantizer_state"]

    replace_quant_module(model)
    set_quantizer_by_cfg(model, config["quant_cfg"])

    unmatched_keys = quantizer_state_dict.keys() - quantizer_state(model).keys()
    extra_keys = quantizer_state(model).keys() - quantizer_state_dict.keys()

    if unmatched_keys:
        raise ApplyModeError(f"Unmatched keys in quantizer state_dict: {unmatched_keys}")
    if extra_keys:
        raise ApplyModeError(f"Extra keys in quantizer state_dict: {extra_keys}")

    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            module.set_from_modelopt_state(quantizer_state_dict[name], name)

    for name, module in model.named_modules():
        if is_quantized_layer_with_weight(module):
            QuantLinearConvBase.initialize_quantizer_with_dummy_states(module)
        if isinstance(module, TensorQuantizer):
            module.clean_up_after_set_from_modelopt_state(name)

    return model


def update_quantize_metadata(
    model: nn.Module, config: QuantizeConfig, metadata: MetadataDict
) -> None:
    """Update the quantizer state in the metadata dict."""
    metadata["quantizer_state"] = quantizer_state(model)


def quantizer_state(model: nn.Module) -> Dict[str, Any]:
    """Returns the quantizer state dict describing the quantizer states in the model."""
    return {
        n: m.get_modelopt_state()
        for n, m in model.named_modules()
        if isinstance(m, (TensorQuantizer, SequentialQuantizer))
    }


def replace_quant_module(model: nn.Module):
    """Recursively replace the module with quantized module."""
    assert not is_quantized(model), "Model must not be quantized!"

    register_custom_model_plugins_on_the_fly(model)

    # If the model has a corresponding quantization module, replace it with it's quantized module
    if type(model) in QuantModuleRegistry:
        model = QuantModuleRegistry.convert(model)

    def _replace_quant_module(model):
        for name, module in model.named_children():
            if type(module) in QuantModuleRegistry:
                setattr(model, name, QuantModuleRegistry.convert(module))
            # Continue replacing in case of nested quantization as well
            _replace_quant_module(getattr(model, name))

    _replace_quant_module(model)

    replaced_modules = sum(isinstance(m, TensorQuantizer) for _, m in model.named_modules())
    print(f"Inserted {replaced_modules} quantizers")


def set_quantizer_by_cfg(quant_model: nn.Module, quant_cfg: Union[QuantizeQuantCfgType, Dict]):
    """Update the quantizer attributes based on the specified `quant_cfg`.

    `quant_cfg` is a dictionary mapping wildcards or filter functions
    to its quantizer attributes which are defined in
    :class:`QuantizerAttributeConfig <.config.QuantizerAttributeConfig>`.
    The wildcards or filter functions  are matched against the quantizer module names.
    The specified quantizer attributes of the matched quantizer modules are set accordingly.
    The key ``"default"`` is a special key that sets the quantizer attributes of all the quantizers for which
    no other wildcard or filter functions match the quantizer module name.

    In addition, the dictionary entries could also be pytorch module class names mapping the class specific
    quantization configuration. The pytorch modules should have a quantized equivalent.

    See :meth:`set_quantizer_attribute <modelopt.torch.quantization.conversion.set_quantizer_attribute>`
    for more details.
    """
    quant_cfg = quant_cfg.copy()
    if "default" in quant_cfg:
        set_quantizer_attribute(quant_model, "*", quant_cfg["default"])
        quant_cfg.pop("default")

    for pattern, cfg in quant_cfg.items():
        if str(pattern) in QuantModuleRegistry:
            parent_class = QuantModuleRegistry[str(pattern)]
            assert isinstance(
                cfg, dict
            ), f"Expected a dictionary for quantizer configuration for child tensor quantizers of {parent_class}."
            for sub_pattern, sub_cfg in cfg.items():
                set_quantizer_attribute(quant_model, sub_pattern, sub_cfg, parent_class)
            continue
        set_quantizer_attribute(quant_model, pattern, cfg)


def set_quantizer_attribute(
    quant_model: nn.Module,
    wildcard_or_filter_func: Union[str, Callable],
    attribute: Union[
        QuantizerAttributeConfig,
        List[QuantizerAttributeConfig],
        Dict[
            Union[str, Callable],
            Union[QuantizerAttributeConfig, List[QuantizerAttributeConfig]],
        ],
        Dict,
        List[Dict],
    ],
    parent_class: Union[None, type] = None,
):
    """Finegrained adjustment of quantizer attribute by wildcard or filter function.

    Args:
        quant_model: A pytorch model
        wildcard_or_filter_func: a wildcard string or a filter function. The wildcard string is matched
            against the quantizer module names. The quantizer modules are
            instances of
            :class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer>`.
            The filter function takes a quantized module name as input and returns ``True`` if the
            quantizer should be adjusted and ``False`` otherwise.
        attribute:  An instance of :class:`QuantizerAttributeConfig <.config.QuantizerAttributeConfig>` or an equivalent
            dictionary or a list of these two types.
            If ``attribute`` is a list, the matched
            :class:`TensorQuantizer <nn.modules.tensor_quantizer.TensorQuantizer>`
            modules will be replaced with :class:`SequentialQuantizer <nn.modules.tensor_quantizer.SequentialQuantizer>`
            modules having one quantizer for each attribute instance from the list.
            See
            :meth:`set_from_attribute_config() <nn.modules.tensor_quantizer.TensorQuantizer.set_from_attribute_config>`
            for more details on the supported attributes and their types.
        parent_class: (Optional) The parent class of the quantizer modules matching ``wildcard_or_filter_func`` which
            should be adjusted. If ``None``, all the matching quantizer modules will be adjusted.
    """
    for name, module in quant_model.named_modules():
        if isinstance(module, TensorQuantizer):
            if isinstance(wildcard_or_filter_func, str):
                if not fnmatch.fnmatch(name, wildcard_or_filter_func):
                    continue
            elif callable(wildcard_or_filter_func):
                if not wildcard_or_filter_func(name):
                    continue
            else:
                raise NotImplementedError(f"Unsupported type {type(wildcard_or_filter_func)}")

            if parent_class is not None and not isinstance(
                quant_model.get_submodule(".".join(name.split(".")[:-1])), parent_class
            ):
                continue

            if isinstance(attribute, list):
                parent_module = quant_model.get_submodule(name.rpartition(".")[0])
                module = SequentialQuantizer(*(TensorQuantizer() for _ in range(len(attribute))))
                setattr(parent_module, name.split(".")[-1], module)

            module.set_from_attribute_config(attribute)


def register(original_cls: nn.Module, quantized_cls: nn.Module):
    """Register a quantized class for the given un-quantized original class.

    Args:
        original_cls: The original un-quantized class.
        quantized_cls: The quantized class. This class should have a `_setup` method which initializes
            various quantizers called in the forward. The forward function of the quantized class should call the
            quantizers at the correct location.

    Here is an example of defining a quantized class and registering it:

    .. code-block:: python

        import modelopt.torch.quantization as mtq
        from modelopt.torch.quantization.nn import TensorQuantizer


        class QuantLayerNorm(nn.LayerNorm):
            def __init__(self, normalized_shape):
                super().__init__(normalized_shape)
                self._setup()

            def _setup(self):
                # Method to setup the quantizers
                self.input_quantizer = TensorQuantizer()
                self.weight_quantizer = TensorQuantizer()

            def forward(self, input):
                input = self.input_quantizer(input)
                weight = self.weight_quantizer(self.weight)
                return F.layer_norm(input, self.normalized_shape, weight, self.bias, self.eps)


        # Register the custom quantized module
        mtq.register(original_cls=nn.LayerNorm, quantized_cls=QuantLayerNorm)

    """
    assert hasattr(
        quantized_cls, "_setup"
    ), "Quantized class must have a _setup method which initializes various quantizers."

    quantized_dm_cls = type(
        quantized_cls.__name__, (quantized_cls, DynamicModule, original_cls), {}
    )

    QuantModuleRegistry.register({original_cls: original_cls.__name__})(quantized_dm_cls)


def unregister(original_cls: nn.Module):
    """Unregister the quantized class for the given un-quantized original class.

    Args:
        original_cls: The original un-quantized class.

    """
    QuantModuleRegistry.unregister(original_cls)


def export_quantized_model(model: nn.Module, config: _QuantizeExportConfig) -> ConvertReturnType:
    """Export the quantized model to a quantized model."""
    raise NotImplementedError("Exporting a quantized model is not supported yet.")


def restore_export_quantized_model(
    model: nn.Module, config: _QuantizeExportConfig, metadata: MetadataDict
) -> nn.Module:
    """Restores the quantized model from the given state dict."""
    raise NotImplementedError("Restoring a quantized & exported model is not supported yet.")
