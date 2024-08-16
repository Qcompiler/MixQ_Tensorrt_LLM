# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Any, Dict, List, Tuple

import numpy as np
import tensorrt as trt
import torch

from modelopt.onnx.utils import get_node_names_from_bytes

from ..utils import OnnxBytes
from .registry import RuntimeRegistry
from .runtime_client import Deployment, DeploymentTable, DetailedResults, RuntimeClient
from .tensorrt.constants import SHA_256_HASH_LENGTH
from .tensorrt.engine_builder import build_engine, profile_engine
from .tensorrt.parse_trtexec_log import parse_profiling_log
from .tensorrt.tensorrt_utils import convert_trt_dtype_to_torch

__all__ = ["TRTLocalClient"]


@RuntimeRegistry.register("TRT")
class TRTLocalClient(RuntimeClient):
    """A client for using the local TRT runtime with GPU backend."""

    @property
    def default_deployment(self) -> Deployment:
        return {k: v[0] for k, v in self.deployment_table.items()}

    @property
    def deployment_table(self) -> DeploymentTable:
        return {
            "version": ["8.6", "9.1", "9.2", "9.3", "10.0"],
            "accelerator": ["GPU"],
            "precision": ["fp32", "fp16", "fp8", "int8", "int4"],
            # Support ONNX opsets 13-19
            "onnx_opset": [str(i) for i in range(13, 20)],
        }

    def __init__(self, deployment: Deployment):
        """Initialize a TRTLocalClient with the given deployment."""
        super().__init__(deployment)
        self.inference_sessions = {}
        logger = trt.Logger(trt.Logger.WARNING)
        self.trt_runtime = trt.Runtime(logger)
        assert trt.init_libnvinfer_plugins(logger, ""), "Failed to initialize nvinfer plugins."
        self.stream = torch.cuda.Stream()

    def _ir_to_compiled(self, ir_bytes: bytes, compilation_args: Dict[str, Any] = None) -> bytes:
        """Converts an ONNX model to a compiled TRT engine.

        Args:
            ir_bytes: The ONNX model bytes.
            compilation_args: A dictionary of compilation arguments. Supported args: dynamic_shapes, plugin_config.

        Returns:
            The compiled TRT engine bytes.
        """
        onnx_bytes = OnnxBytes.from_bytes(ir_bytes)
        onnx_model_file_bytes = onnx_bytes.get_onnx_model_file_bytes()
        self.node_names = get_node_names_from_bytes(onnx_model_file_bytes)
        engine_bytes, _, _ = build_engine(
            onnx_bytes,
            dynamic_shapes=compilation_args.get("dynamic_shapes", None),
            plugin_config=compilation_args.get("plugin_config", None),
            trt_mode=self.deployment["precision"],
        )
        self.engine_bytes = engine_bytes
        return engine_bytes

    def _profile(self, compiled_model: bytes) -> Tuple[float, DetailedResults]:
        _, trtexec_log = profile_engine(
            compiled_model, self.node_names, enable_layerwise_profiling=True
        )
        profiling_results = parse_profiling_log(trtexec_log.decode())
        latency = 0.0
        detailed_results = {}
        if profiling_results is not None:
            latency = profiling_results["performance_summary"]["Latency"][1]
            detailed_results = profiling_results
        return latency, detailed_results

    def _inference(self, compiled_model: bytes, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Run inference with the compiled model and return the output as list of numpy arrays."""
        assert compiled_model is not None, "Engine bytes are not set."

        model_hash = compiled_model[:SHA_256_HASH_LENGTH]
        if model_hash not in self.inference_sessions:
            model_bytes = compiled_model[SHA_256_HASH_LENGTH:]
            self.inference_sessions[model_hash] = self.TRTSession(
                model_bytes, self.trt_runtime, self.stream
            )
        return self.inference_sessions[model_hash].run(inputs)

    def _teardown_all_sessions(self):
        """Clean up all TRT sessions."""
        for session in self.inference_sessions.values():
            del session
        self.inference_sessions = {}

    class TRTSession:
        def __init__(self, compiled_model, trt_runtime, stream):
            self.engine = trt_runtime.deserialize_cuda_engine(compiled_model)
            assert self.engine is not None, "Engine deserialization failed."
            self.execution_context = self.engine.create_execution_context()
            self.stream = stream
            self.input_tensors, self.output_tensors = self.initialize_input_output_tensors(
                self.engine
            )

        def initialize_input_output_tensors(self, engine):
            # Allocate torch tensors for inputs and outputs
            input_tensors = []
            output_tensors = []
            for idx in range(engine.num_io_tensors):
                tensor_name = engine.get_tensor_name(idx)
                tensor_shape = engine.get_tensor_profile_shape(tensor_name, 0)[0]
                tensor_dtype = convert_trt_dtype_to_torch(engine.get_tensor_dtype(tensor_name))
                torch_tensor = torch.empty(tuple(tensor_shape), dtype=tensor_dtype, device="cuda")
                if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    input_tensors.append(torch_tensor)
                    self.execution_context.set_tensor_address(
                        tensor_name, input_tensors[idx].data_ptr()
                    )
                else:
                    output_tensors.append(torch_tensor)
                    self.execution_context.set_tensor_address(
                        tensor_name,
                        output_tensors[idx - len(input_tensors)].data_ptr(),
                    )
            assert (
                self.execution_context.all_shape_inputs_specified
            ), "Not all shape inputs are specified."

            # Set selected profile idx
            self.execution_context.set_optimization_profile_async(0, self.stream.cuda_stream)

            # Assertion: to ensure all the inputs are set
            assert (
                len(self.execution_context.infer_shapes()) == 0
            ), "Shapes of all the bindings cannot be inferred."

            return input_tensors, output_tensors

        def run(self, inputs):
            assert self.engine is not None, "Engine is not set."

            # Copy inputs to GPU
            with torch.cuda.stream(self.stream):
                for i, input_np in enumerate(inputs):
                    input_t = torch.from_numpy(input_np)

                    # Pad the input tensor with zeros if the input tensor is smaller than the expected size
                    zero_tensor = torch.zeros(self.input_tensors[i].shape)
                    slices = tuple(slice(0, input_t.size(dim)) for dim in range(input_t.dim()))
                    zero_tensor[slices] = input_t

                    # Copy the input tensor to the GPU
                    self.input_tensors[i].copy_(zero_tensor, non_blocking=True)

            # Run inference
            self.execution_context.execute_async_v3(stream_handle=self.stream.cuda_stream)

            # Copy outputs to CPU
            with torch.cuda.stream(self.stream):
                for t in self.output_tensors:
                    t.to(device="cpu", non_blocking=True)

            self.stream.synchronize()

            return [t.detach().cpu().numpy() for t in self.output_tensors]
