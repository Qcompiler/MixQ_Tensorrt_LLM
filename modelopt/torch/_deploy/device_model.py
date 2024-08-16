# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Class representing the compiled model for a particular device."""
import os
from typing import Any, Tuple

from modelopt.torch.utils import numpy_to_torch, torch_to_numpy, unflatten_tree

from ._runtime import DetailedResults, RuntimeClient
from .utils import ModelMetadata, generate_onnx_input


class DeviceModel:
    """On-device model with profiling functions and PyTorch-like inference interface.

    This object should be generated from
    :meth:`compile <modelopt.torch._deploy.compilation.compile>`.
    """

    def __init__(self, client: RuntimeClient, compiled_model: bytes, metadata: ModelMetadata):
        """Initialize a device model with the corresponding model, onnx, and engine model.

        Args:
            client: the runtime client used to compile the model.
            compiled_model: Compiled device model created during runtime compilation.
            metadata: The model's metadata (needed for inference/profiling with compiled model).
        """
        self.client = client
        self.compiled_model = compiled_model
        self.model_metadata = metadata

    def __call__(self, *args, **kwargs):
        """Execute forward function of the model on the specified deployment and return output."""
        return self.forward(*args, **kwargs)

    def get_latency(self) -> float:
        """Profiling API to let user get model latency with the compiled device model.

        Returns:
            The latency of the compiled model in ms.
        """
        latency, _ = self._profile_device()
        return latency

    def profile(self, verbose: bool = False) -> Tuple[float, DetailedResults]:
        """Inference API to let user do inference with the compiled device model.

        Args:
            verbose: If True, print out the profiling results as a table.

        Returns: A tuple (latency, detailed_result) where
            ``latency`` is the latency of the compiled model in ms,
            ``detailed_result`` is a dictionary containing additional benchmarking results
        """
        latency, detailed_result = self._profile_device()

        if verbose:
            print(detailed_result)

        return latency, detailed_result

    def forward(self, *args, **kwargs) -> Any:
        """Execute forward of the model on the specified deployment and return output.

        Arguments:
            args: Non-keyword arguments to the model for inference.
            kwargs: Keyword arguments to the model for inference.

        Returns:
            The inference result in the same (nested) data structure as the original model.

        .. note::

            This API let the users do inference with the compiled device model.

        .. warning::

            All return values will be of type ``torch.Tensor`` even if the original model returned
            native python types such as bool/int/float.
        """
        if self.compiled_model is None:
            raise AttributeError("Please compile the model first.")

        # Flatten all args, kwargs into a single list of tensors for onnx/device inference.
        all_args = args + (kwargs,) if kwargs or (args and isinstance(args[-1], dict)) else args

        # If Model metadata is None then DeviceModel is instantiated with raw ONNX bytes instead of PyTorch module.
        onnx_inputs = all_args[0]
        if self.model_metadata:
            onnx_inputs = list(generate_onnx_input(self.model_metadata, all_args).values())

        # run inference with the engine equivalent of the model
        np_inputs = torch_to_numpy(onnx_inputs)
        np_outputs = self.client.inference(compiled_model=self.compiled_model, inputs=np_inputs)
        onnx_outputs = numpy_to_torch(np_outputs)

        # Note that bool/float/ints will be returned as corresponding tensors
        # TODO: maybe eventually we want to compare this against the original types
        # generate expected returned data structure of the model
        if not self.model_metadata:
            return onnx_outputs
        return unflatten_tree(onnx_outputs, self.model_metadata["output_tree_spec"])

    def save_compile_model(self, path: str, remove_hash: bool = False):
        """Saves the compiled model to a file.

        Args:
            path: The path to save the compiled model.
            remove_hash: If True, the hash will be removed from the saved model.
        """
        compiled_model = self.compiled_model
        if remove_hash:
            compiled_model = compiled_model[32:]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(compiled_model)

    def _profile_device(self) -> Tuple[float, DetailedResults]:
        """Profile the device model stored in self and return latency results."""
        return self.client.profile(compiled_model=self.compiled_model)
