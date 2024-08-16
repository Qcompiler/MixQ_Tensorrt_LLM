# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""User-facing quantization API."""
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn

from modelopt.torch.opt import apply_mode
from modelopt.torch.opt.searcher import ForwardLoop

from .algorithms import AutoQuantizeSearcher
from .conversion import set_quantizer_attribute
from .mode import QuantizeModeRegistry
from .model_calib import calibrate
from .nn import TensorQuantizer

__all__ = [
    "quantize",
    "auto_quantize",
    "disable_quantizer",
    "enable_quantizer",
    "print_quant_summary",
    "fold_weight",
]


def quantize(
    model: nn.Module,
    config: Dict[str, Any],
    forward_loop: Optional[ForwardLoop] = None,
) -> nn.Module:
    """Quantizes and calibrates the model in-place.

    This method performs replacement of modules with their quantized counterparts and
    performs calibration as specified by ``quant_cfg``.
    ``forward_loop`` is used to forward data through the model and gather statistics for calibration.

    Args:
        model: A pytorch model
        config: A dictionary or an instance of
            :class:`QuantizeConfig <modelopt.torch.quantization.config.QuantizeConfig>` specifying the
            values for keys ``"quant_cfg"`` and ``"algorithm"``.
            It is basically a dictionary specifying the values for keys ``"quant_cfg"`` and ``"algorithm"``.
            The ``"quant_cfg"`` key specifies the quantization configurations.
            The ``"algorithm"`` key specifies the ``algorithm`` argument to
            :meth:`calibrate <modelopt.torch.quantization.model_calib.calibrate>`.

            Quantization configurations is a dictionary mapping wildcards or filter functions
            to its quantizer attributes. The wildcards or filter functions  are matched
            against the quantizer module names. The quantizer modules have names ending with
            ``weight_quantizer`` and ``input_quantizer`` and they perform weight quantization and
            input quantization (or activation quantization) respectively. The quantizer modules
            are instances of
            :class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer>`.
            The quantizer attributes are defined by :class:`QuantizerAttributeConfig`. See
            :class:`QuantizerAttributeConfig` for details on the quantizer attributes and their values.

            An example ``config`` dictionary is given below:

            .. code-block::python

                config = {

                    "quant_cfg": {
                        # "num_bits" specifies the number of bits for quantization
                        # "axis" specifies the axis for quantization
                        "*weight_quantizer": {"num_bits": 8, "axis": 0},
                        "*input_quantizer": {"num_bits": 8, "axis": -1},

                        # Default quantization settings
                        "default": {"num_bits": 8, "axis": None},
                    }
                    "algorithm": "max"
                }

            See :ref:`Quantization Formats <quantization-formats>` to learn more about the supported
            quantization formats. See :ref:`Quantization Configs <quantization-configs>` for more details on
            ``config`` dictionary.

        forward_loop: A callable that forwards all calibration data through the model. This is used
            to gather statistics for calibration. It should take model as the argument. It does not need
            to return anything.

            This argument is not required for weight-only quantization with the ``"max"``
            algorithm.

            Here are a few examples for correct ``forward_loop`` definitions:
            Example 1:

            .. code-block::

                    def forward_loop(model) -> None:
                        # iterate over the data loader and forward data through the model
                        for batch in data_loader:
                            model(batch)

            Example 2:

            .. code-block::

                    def forward_loop(model) -> float:
                        # evaluate the model on the task
                        return evaluate(model, task, ....)

            Example 3:

            .. code-block::

                    def forward_loop(model) -> None:
                        # run evaluation pipeline
                        evaluator.model = model
                        evaluator.evaluate()

            .. note::

                Calibration does not require forwarding the entire dataset through the model.
                Please subsample the dataset or reduce the number of batches if needed.

    Returns: A pytorch model which has been quantized and calibrated.
    """
    model = apply_mode(model, mode=[("quantize", config)], registry=QuantizeModeRegistry)
    return calibrate(model, config["algorithm"], forward_loop=forward_loop)


def auto_quantize(
    model: nn.Module,
    data_loader: Iterable,
    loss_func: Callable[[Any, Any], torch.Tensor],
    constraints: Dict[str, Union[float, str]] = {"weight_compression": 0.30},
    quantization_formats: List[Optional[str]] = ["FP8_DEFAULT_CFG", None],
    collect_func: Optional[Callable] = None,
    num_calib_steps: int = 512,
    num_score_steps: int = 128,
    verbose: bool = False,
):
    r"""Quantizes a model by searching for the best quantization formats per-layer.

    ``auto_quantize`` uses a gradient based sensitivity score to rank the per-layer quantization formats and search
    for the best quantization formats per-layer.

    Args:
        model: A pytorch model with quantizer modules.
        data_loader: An iterator that yields data that is to be used for calibrating quantized layers and estimating
            ``auto_quantize`` scores.
        loss_func: A ``Callable`` which takes the model output (i.e output of ``model.forward()``)
              and the batch of data as its inputs and returns a scalar loss.

              It should be possible to run a backward pass on the loss value returned by this method.

              ``collect_func`` will be used to gather the inputs to ``model.forward()``
              from a batch of data yielded by``data_loader``.

              ``loss_func`` should support the following usage:

              .. code-block:: python

                    for batch in data_loader:
                        # Assuming collect_func returns a tuple of arguments
                        output = model(*collect_func(batch))

                        loss = loss_func(output, batch)
                        loss.backward()

        constraints: Constraints for the search. Currently we support ``weight_compression``.
            For example, for a compressed weight of 0.30 (i.e, 30%) of the original total weight size,
            ``constraints`` should be:

            .. code-block:: python

                # For a compressed weight of 0.30 (i.e, 30%) of the original total weight size
                constraints = {"weight_compression": 0.30}

            You can also provide the equivalent percentage value in string type. For example:

            .. code-block:: python

                # For a compressed weight of 30% of the original total weight size
                constraints = {"weight_compression": "30%"}

        quantization_formats: A list of the string names of the quantization formats to search for.
            The supported quantization formats are as listed by :attr:`modelopt.torch.quantization.config.choices`.

            In addition, the quantization format can also be ``None`` which implies skipping quantization for
            the layer.

            .. note::

                The quantization formats will be applied on a per-layer match basis. The global model level name
                based quantizer attribute setting will be ignored. For example, in ``FP8_DEFAULT_CFG`` quantizer
                configuration the key ``"*lm_head*": {"enable": False}`` disables quantization for the ``lm_head``
                layer. However in ``auto_quantize``, the quantization format for the ``lm_head`` layer will be searched.
                This is because the key ``"*lm_head*"`` sets the quantizer attributes based on the global model level
                name, not per-layer basis. The keys ``"*input_quantizer"``, ``"*weight_quantizer"`` etc. in
                ``FP8_DEFAULT_CFG`` match on a per-layer basis  - hence the corresponding quantizers
                will be set as specified.

        collect_func: A ``Callable`` that takes a batch of data from the data loader as input and returns the input to
            ``model.forward()`` as described in
            :meth:`run_forward_loop <modelopt.torch.utils.network.run_forward_loop>`.
        num_calib_steps: Number of batches to use for calibrating the quantized model. Suggested value is 512.
        num_score_steps: Number of batches to use for estimating ``auto_quantize`` scores. Suggested value is 128.
            A higher value could increase the time taken for performing ``auto_quantize``.
        verbose: If True, prints the search progress/intermediate results.

    Returns: A tuple (model, state_dict) where ``model`` is the searched and quantized model and
        ``state_dict`` contains the history and detailed stats of the search procedure.

    .. note::

        ``auto_quantize`` groups certain layers and restricts the quantization formats for them to be same. For example,
        Q, K, V linear layers belonging to the same transformer layer will have the same quantization format.
        This is to ensure compatibility with TensorRT-LLM which fuses these three linear layers into a single linear
        layer.

        A list of regex pattern rules as defined in :attr:`rules <.algorithms.AutoQuantizeSearcher.rules>`
        are used to specify the group of layers. The first captured group
        in the regex pattern (i.e, ``pattern.match(name).group(1)``) is used to group the layers. All the layers
        that share the same first captured group will have the same quantization format..

        For example, the rule ``r"^(.*?)\.(q_proj|k_proj|v_proj)$"``
        groups the `q_proj`, `k_proj`, `v_proj` linear layers belonging to the same transformer layer.

        You may modify the rules to group the layers as per your requirement.

        .. code-block:: python

            from modelopt.torch.quantization.algorithms import AutoQuantizeSearcher

            # To additionally group the layers belonging to same `mlp` layer,
            # add the following rule
            AutoQuantizeSearcher.rules.append(r"^(.*?)\.mlp")

            # Perform `auto_quantize`
            model, state_dict = auto_quantize(model, ...)

    .. note::

        The ``auto_quantize`` API and algorithm is experimental and subject to change. ``auto_quantize`` searched models
        might not be readily deployable to TensorRT-LLM yet.

    """
    model = apply_mode(
        model,
        mode=[("quantize", {"quant_cfg": {}})],
        registry=QuantizeModeRegistry,
    )
    searcher = AutoQuantizeSearcher()
    search_config = {
        "data_loader": data_loader,
        "loss_func": loss_func,
        "quantization_formats": quantization_formats,
        "collect_func": collect_func,
        "num_calib_steps": num_calib_steps,
        "num_score_steps": num_score_steps,
        "verbose": verbose,
    }
    searcher.search(model, constraints, config=search_config)  # type: ignore[arg-type]

    return model, searcher.state_dict()


def disable_quantizer(model: nn.Module, wildcard_or_filter_func: Union[str, Callable]):
    """Disable quantizer by wildcard or filter function."""
    set_quantizer_attribute(model, wildcard_or_filter_func, {"enable": False})


def enable_quantizer(model: nn.Module, wildcard_or_filter_func: Union[str, Callable]):
    """Enable quantizer by wildcard or filter function."""
    set_quantizer_attribute(model, wildcard_or_filter_func, {"enable": True})


def print_quant_summary(model: nn.Module):
    """Print summary of all quantizer modules in the model."""
    count = 0
    for name, mod in model.named_modules():
        if isinstance(mod, TensorQuantizer):
            print(f"{name:80} {mod}")
            count += 1
    print(f"{count} TensorQuantizers found in model")


def fold_weight(model: nn.Module):
    """Fold weight quantizer for fast evaluation."""
    for name, module in model.named_modules():
        if hasattr(module, "weight_quantizer") and hasattr(module, "weight"):
            module.weight.data.copy_(
                (module.weight_quantizer(module.weight.float())).to(module.weight.dtype)
            )
            module.weight_quantizer.disable()
