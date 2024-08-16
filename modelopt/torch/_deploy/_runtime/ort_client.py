# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort

from .registry import RuntimeRegistry
from .runtime_client import Deployment, DeploymentTable, DetailedResults, RuntimeClient

__all__ = ["ORTLocalClient"]


@RuntimeRegistry.register("ORT")
class ORTLocalClient(RuntimeClient):
    """A client for using the local onnx runtime with CPU backend."""

    @property
    def _profile_defaults(self) -> Dict[str, Union[int, float]]:
        """Default profiling parameters."""
        return {
            "iterations": 10,
            "iterations_max": 1000,
            "warm_up": 0.2,
            "duration": 3.0,
        }

    @property
    def _accelerator_to_provider(self) -> Dict[str, str]:
        """Maps accelerator to ORT execution provider."""
        return {
            "CPU": "CPUExecutionProvider",
        }

    @property
    def default_deployment(self) -> Deployment:
        return {k: v[0] for k, v in self.deployment_table.items()}

    @property
    def deployment_table(self) -> DeploymentTable:
        return {
            "version": ["1.16"],
            "accelerator": list(self._accelerator_to_provider.keys()),
            "precision": ["fp32"],
            "onnx_opset": ["13"],
        }

    def _ir_to_compiled(self, ir_bytes: bytes, compilation_args: Dict[str, Any] = None) -> bytes:
        """Converts an ONNX model to a compiled device model."""
        return ir_bytes  # ir_bytes (onnx) are also compiled model for ORT

    def _onnx_to_np_dtype(self, onnx_type: str) -> np.dtype:
        """Maps an ONNX data type to a numpy data type."""
        return {
            "tensor(float16)": np.float16,
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(int8)": np.int8,
            "tensor(int16)": np.int16,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
            "tensor(uint8)": np.uint8,
            "tensor(uint16)": np.uint16,
            "tensor(uint32)": np.uint32,
            "tensor(uint64)": np.uint64,
        }[onnx_type]

    def _init_session(
        self, compiled_model: bytes, session_options: Optional[ort.SessionOptions] = None
    ) -> ort.InferenceSession:
        """Initializes an inference session with the compiled model and returns the session."""
        provider = self._accelerator_to_provider[self.deployment["accelerator"]]
        return ort.InferenceSession(compiled_model, session_options, providers=[provider])

    def _profile(self, compiled_model: bytes) -> Tuple[float, DetailedResults]:
        """Profiles a compiled device model and returns the latency & detailed profiling results."""
        # use a temp folder for profiling results
        with tempfile.TemporaryDirectory() as temp_dir:
            # initialize session + options
            session_options = ort.SessionOptions()
            session_options.enable_profiling = True
            session_options.profile_file_prefix = os.path.join(temp_dir, "ort_profile")
            ort_session = self._init_session(compiled_model, session_options)

            # generate dummy inputs from ort_session.get_inputs()
            inputs = [
                np.asarray(np.random.rand(*x.shape)).astype(self._onnx_to_np_dtype(x.type))
                for x in ort_session.get_inputs()
            ]

            # run session with dummy inputs
            self._run_session(ort_session, inputs, **self._profile_defaults)  # type: ignore[arg-type]

            # end profiling and load results
            prof_file = ort_session.end_profiling()
            with open(prof_file, "r") as p_file:
                results = json.load(p_file)

        # get latency from profiling results (latencies are in nano-seconds)
        # We generally use milliseconds for latency, so divide by 1e3
        latencies = [x["dur"] for x in results if x.get("name") == "model_run"]
        avg_latency = np.mean(latencies) / 1e3

        # return latency & detailed profiling results
        return avg_latency, {"ort_results": results}

    def _inference(self, compiled_model: bytes, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Run inference with the compiled model and return the output as list of numpy arrays."""
        # initialize session, run session, and return session outputs
        ort_session = self._init_session(compiled_model)
        return self._run_session(ort_session, inputs)

    def _run_session(
        self,
        ort_session: ort.InferenceSession,
        inputs: List[np.ndarray],
        iterations: int = 1,
        iterations_max: int = 1000,
        warm_up: float = 0.0,
        duration: float = 0.0,
    ) -> List[np.ndarray]:
        """Run inference with the compiled model and return the output as list of numpy arrays.

        Args:
            ort_session: The ONNX runtime inference session.
            inputs: The input tensors to the session.
            iterations: The minimum number of iterations to run the session for.
            iterations_max: The maximum number of iterations to run the session for.
            warm_up: The number of seconds to warm up the session for.
            duration: The minimum number of seconds to run the session for.

        Returns:
            The outputs of the session as a list of numpy arrays.
        """
        assert iterations > 0, "Number of iterations must be positive!"

        # basics of the session
        ort_inputs = {ort_session.get_inputs()[i].name: inp for i, inp in enumerate(inputs)}
        time_elapsed = 0.0

        # small utility for running the session once with time measurement
        def _run_session_once():
            nonlocal time_elapsed
            time_elapsed -= time.perf_counter()
            ort_outputs = ort_session.run(None, ort_inputs)
            time_elapsed += time.perf_counter()
            return ort_outputs

        # warm-up
        while time_elapsed < warm_up:
            _run_session_once()

        # run inference
        time_elapsed = 0.0
        iter = 0
        total_time = -time.time()
        while (iter < iterations or time_elapsed < duration) and iter < iterations_max:
            ort_outputs = _run_session_once()
            iter += 1
        total_time += time.time()
        print(f"ORT: {iter} iterations in {total_time:.3f} seconds ({total_time/iter:.3f} s/iter)")

        # return session outputs of final iteration
        return ort_outputs
