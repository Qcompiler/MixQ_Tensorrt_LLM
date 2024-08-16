# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Module to profile a model on a target device."""
from typing import Any, Tuple, Union

import torch.nn as nn

from ._runtime import Deployment, DetailedResults
from .compilation import compile

__all__ = ["get_latency", "profile"]


def get_latency(
    model: nn.Module,
    dummy_input: Union[Any, Tuple],
    deployment: Deployment,
) -> float:
    """Obtain deployment latency of model by compiling and sending model to engine for profiling.

    Args:
        model: PyTorch model to compile for target device.
        dummy_input: Arguments of ``model.forward()``. This is used for exporting and calculating
            inference-based metrics, such as latency/FLOPs. The format of ``dummy_inputs`` follows
            the convention of the ``args`` argument in
            `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_.
            Specifically, ``dummy_input`` can be:

            #. a single argument (``type(dummy_input) != tuple``) corresponding to

               .. code-block:: python

                    model.forward(dummy_input)

            #. a tuple of arguments corresponding to

               .. code-block:: python

                    model.forward(*dummy_input)

            #. a tuple of arguments such that ``type(dummy_input[-1]) == dict`` corresponding to

               .. code-block:: python

                    model.forward(*dummy_input[:-1], **dummy_input[-1])

               .. warning::

                   In this case the model's ``forward()`` method **cannot** contain keyword-only
                   arguments (e.g. ``forward(..., *, kw_only_args)``) or variable keyword arguments
                   (e.g. ``forward(..., **kwargs)``) since these cannot be sorted into positional
                   arguments.

            .. note::

                In order to pass a dict as last non-keyword argument, you need to use a tuple as
                ``dummy_input`` and add an *empty* dict as the last element, e.g.,

                .. code-block:: python

                    dummy_input = (x, {"y": y, "z": z}, {})

                The empty dict at the end will then be interpreted as the keyword args.

            See `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_
            for more info.

            Note that if you provide a ``{arg_name}`` with batch size ``b``, the results will be
            computed based on batch size ``b``.

        deployment: Deployment configuration with keys as specified below.

            * ``runtime``: the desired runtime for deployment (*required*);
            * ``accelerator``: the accelerator on the device to be used (*optional*);
            * ``version``: the version of runtime to be used (*optional*);
            * ``precision``: the desired precision (*optional*);
            * ``onnx_opset``: the opset version (*optional*).

            An example of a deployment configuration is:

            .. code-block:: python

                deployment = {
                    "runtime": "ORT",
                    "accelerator": "CPU",
                    "version": "1.11",
                    "precision": "fp32",
                    "onnx_opset": 13,
                }

    Returns:
        The latency of the compiled model in ms.
    """
    device_model = compile(model, dummy_input, deployment)
    return device_model.get_latency()


def profile(
    model: nn.Module,
    dummy_input: Union[Any, Tuple],
    deployment: Deployment,
    verbose: bool = False,
) -> Tuple[float, DetailedResults]:
    """Model profile method to help user to profile their model on target device.

    Args:
        model: PyTorch model to compile for target device.
        dummy_input: Arguments of ``model.forward()``. This is used for exporting and calculating
            inference-based metrics, such as latency/FLOPs. The format of ``dummy_inputs`` follows
            the convention of the ``args`` argument in
            `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_.
            Specifically, ``dummy_input`` can be:

            #. a single argument (``type(dummy_input) != tuple``) corresponding to

               .. code-block:: python

                    model.forward(dummy_input)

            #. a tuple of arguments corresponding to

               .. code-block:: python

                    model.forward(*dummy_input)

            #. a tuple of arguments such that ``type(dummy_input[-1]) == dict`` corresponding to

               .. code-block:: python

                    model.forward(*dummy_input[:-1], **dummy_input[-1])

               .. warning::

                   In this case the model's ``forward()`` method **cannot** contain keyword-only
                   arguments (e.g. ``forward(..., *, kw_only_args)``) or variable keyword arguments
                   (e.g. ``forward(..., **kwargs)``) since these cannot be sorted into positional
                   arguments.

            .. note::

                In order to pass a dict as last non-keyword argument, you need to use a tuple as
                ``dummy_input`` and add an *empty* dict as the last element, e.g.,

                .. code-block:: python

                    dummy_input = (x, {"y": y, "z": z}, {})

                The empty dict at the end will then be interpreted as the keyword args.

            See `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_
            for more info.

            Note that if you provide a ``{arg_name}`` with batch size ``b``, the results will be
            computed based on batch size ``b``.
        deployment: Deployment configuration with keys as specified below.

            * ``runtime``: the desired runtime for deployment (*required*);
            * ``accelerator``: the accelerator on the device to be used (*optional*);
            * ``version``: the version of runtime to be used (*optional*);
            * ``precision``: the desired precision (*optional*);
            * ``onnx_opset``: the opset version (*optional*).

            An example of a deployment configuration is:

            .. code-block:: python

                deployment = {
                    "runtime": "ORT",
                    "accelerator": "CPU",
                    "version": "1.11",
                    "precision": "fp32",
                    "onnx_opset": 13,
                }
        verbose: If True, print out the profiling results as a table.

        Returns: A tuple (latency, detailed_result) where
            ``latency`` is the latency of the compiled model in ms,
            ``detailed_result`` is a dictionary containing additional benchmarking results
    """
    device_model = compile(model, dummy_input, deployment)
    return device_model.profile(verbose)
