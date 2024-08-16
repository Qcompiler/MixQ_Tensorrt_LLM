# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

__all__ = ["Deployment", "DeploymentTable", "DetailedResults", "RuntimeClient"]

Deployment = Dict[str, str]
DeploymentTable = Dict[str, List[str]]
DetailedResults = Dict[str, Any]


class RuntimeClient(ABC):
    """A abstract client class for implementing various runtimes to be used for deployment.

    The RuntimeClient defines a common interfaces for accessing various runtimes within modelopt.
    """

    _runtime: str  # runtime of the client --> set by RuntimeRegistry.register

    def __init__(self, deployment: Deployment):
        super().__init__()
        self.deployment = self.sanitize_deployment_config(deployment)

    @property
    def runtime(self) -> str:
        return self._runtime

    def sanitize_deployment_config(self, deployment: Deployment) -> Deployment:
        """Cleans/checks the deployment config & fills in runtime-specific default values.

        Args:
            deployment: Deployment config with at least the ``runtime`` key specified.

        Returns:
            The sanitized deployment config with all runtime-specific default values filled
            in for missing keys.
        """
        # check runtime
        assert self.runtime == deployment["runtime"], "Runtime mismatch!"

        # check that version was provided
        if "version" not in deployment:
            raise KeyError("Runtime version must be provided!")

        # fill in default values and update
        deployment = {**self.default_deployment, **deployment}

        # sanity check on keys (inverse doesn't have to be checked since we fill in defaults)
        table = self.deployment_table
        extra_keys = deployment.keys() - table.keys() - {"runtime"}
        assert not extra_keys, f"Invalid deployment config keys detected: {extra_keys}!"

        # sanity checks on values
        invalid_values = {(k, deployment[k]): t for k, t in table.items() if deployment[k] not in t}
        assert not invalid_values, f"Invalid deployment config values detected: {invalid_values}!"

        return deployment

    def ir_to_compiled(self, ir_bytes: bytes, compilation_args: Dict[str, Any] = {}) -> bytes:
        """Converts a model from its intermediate representation (IR) to a compiled device model.
        Args:
            ir_bytes: Intermediate representation (IR) of the model.
            compilation_args: Additional arguments for the compilation process.

        Returns: The compiled device model that can be used for further downstream tasks such as
            on-device inference and profiling,
        """
        # run model compilation
        compiled_model = self._ir_to_compiled(ir_bytes, compilation_args)
        assert compiled_model, "Device conversion failed!"

        return compiled_model

    def profile(self, compiled_model: bytes) -> Tuple[float, DetailedResults]:
        """Profiles a compiled device model and returns the latency & detailed profiling results.

        Args:
            compiled_model: Compiled device model from compilation service.

        Returns: A tuple (latency, detailed_result) where
            ``latency`` is the latency of the compiled model in ms,
            ``detailed_result`` is a dictionary containing additional benchmarking results
        """
        # get latency & detailed results from client
        latency, detailed_result = self._profile(compiled_model)
        assert latency > 0.0, "Profiling failed!"

        return latency, detailed_result

    def inference(self, compiled_model: bytes, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Run inference with the compiled model and return the output as list of numpy arrays.

        Args:
            compiled_model: Compiled device model from compilation service.
            inputs: Inputs to do inference on server.

        Returns:
            A list of torch tensors from the inference outputs
        """
        # run inference
        outputs = self._inference(compiled_model, inputs)
        assert len(outputs) > 0, "Inference failed!"

        return outputs

    @property
    @abstractmethod
    def default_deployment(self) -> Deployment:
        """Provides the default deployment config without the device key."""
        raise NotImplementedError

    @property
    @abstractmethod
    def deployment_table(self) -> DeploymentTable:
        """Provides a set of supported values for each deployment config key."""
        raise NotImplementedError

    @abstractmethod
    def _ir_to_compiled(self, ir_bytes: bytes, compilation_args: Dict[str, Any] = {}) -> bytes:
        """Converts a model from its intermediate representation (IR) to a compiled device model."""

    @abstractmethod
    def _profile(self, compiled_model: bytes) -> Tuple[float, DetailedResults]:
        """Profiles a compiled device model and returns the latency & detailed profiling results."""

    @abstractmethod
    def _inference(self, compiled_model: bytes, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Run inference with the compiled model and return the output as list of numpy arrays."""
