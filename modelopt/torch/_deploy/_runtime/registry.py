# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Callable, Dict, Type

from .runtime_client import Deployment, RuntimeClient

__all__ = ["RuntimeRegistry"]


class RuntimeRegistry:
    """Registry to store and retrieve various runtime client implementations."""

    _runtime_client_lookup: Dict[str, Type[RuntimeClient]] = {}

    @classmethod
    def register(cls, runtime: str) -> Callable[[Type[RuntimeClient]], Type[RuntimeClient]]:
        """A decorator to register a RuntimeClient with its relevant runtime.

        For example:

        .. code-block:: python

            @RuntimeRegistry.register("my_runtime")
            class MyRuntimeClient(RuntimeClient):
                ...
        """

        def _register_runtime_client(new_type: Type[RuntimeClient]) -> Type[RuntimeClient]:
            cls._runtime_client_lookup[runtime] = new_type
            new_type._runtime = runtime
            return new_type

        return _register_runtime_client

    @classmethod
    def unregister(cls, runtime: str) -> None:
        """A helper to unregister a RuntimeClient

        For example:

        .. code-block:: python

            @RuntimeRegistry.register("my_runtime")
            class MyRuntimeClient(RuntimeClient):
                ...


            # later
            RuntimeRegistry.unregister("my_runtime")
        """

        cls._runtime_client_lookup.pop(runtime)

    @classmethod
    def get(cls, deployment: Deployment) -> RuntimeClient:
        """Get the runtime client for the given deployment.

        Args:
            deployment: Deployment configuration.

        Returns:
            The runtime client for the given deployment.
        """
        # check for valid runtime
        if "runtime" not in deployment:
            raise KeyError("Runtime was not set.")
        if deployment["runtime"] not in cls._runtime_client_lookup:
            raise ValueError(f"Runtime {deployment['runtime']} is not supported.")

        # initialize runtime client
        return cls._runtime_client_lookup[deployment["runtime"]](deployment)
