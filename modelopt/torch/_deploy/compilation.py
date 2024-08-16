# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Module to compile a model for a target device."""
from typing import Any, Tuple, Union

import torch.nn as nn

from modelopt.torch.utils import is_channels_last

from ._runtime import Deployment, RuntimeRegistry
from .device_model import DeviceModel
from .utils import OnnxBytes, get_onnx_bytes_and_metadata

__all__ = ["compile"]


def compile(
    model: nn.Module, dummy_input: Union[Any, Tuple], deployment: Deployment
) -> DeviceModel:
    """Compile a given torch model into a device model according to the provided deployment.

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
        An instance of DeviceModel.
    """
    # Add check for nhwc format before we formally support NHWC models
    assert not is_channels_last(model), "Only NCHW models are supported!"

    # Export onnx model and get some required names from it
    onnx_bytes, metadata = get_onnx_bytes_and_metadata(model, dummy_input, "", True, **deployment)

    client = RuntimeRegistry.get(deployment)

    # For the ORTLocalClient, we need to pass the bytes of the onnx model instead of the OnnxBytes object
    if client.__class__.__name__ == "ORTLocalClient":
        onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
        onnx_bytes = onnx_bytes_obj.onnx_model[f"{onnx_bytes_obj.model_name}.onnx"]

    compiled_model = client.ir_to_compiled(onnx_bytes)

    return DeviceModel(client, compiled_model, metadata)
