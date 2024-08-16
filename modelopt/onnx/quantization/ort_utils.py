# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Provides basic ORT inference utils, shoule be replaced by modelopt.torch.ort_client."""

from typing import List

import onnxruntime as ort
from onnxruntime.quantization.operators.qdq_base_operator import QDQOperatorBase
from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry

from modelopt.onnx.quantization.operators import QDQConvTranspose, QDQNormalization
from modelopt.onnx.quantization.ort_patching import patch_ort_modules


def create_inference_session(onnx_path: str):
    """Create an OnnxRuntime InferenceSession."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
    )


def get_quantizable_op_types(op_types_to_quantize: List[str]) -> List[str]:
    """Returns a set of quantizable op types.

    Note. This function should be called after quantize._configure_ort() is called once.
    This returns quantizable op types either from the user supplied parameter
    or from modelopt.onnx's default quantizable ops setting.
    """
    op_types_to_quantize = op_types_to_quantize or []

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        q_linear_ops = list(QLinearOpsRegistry.keys())
        qdq_ops = list(QDQRegistry.keys())
        op_types_to_quantize = list(set(q_linear_ops + qdq_ops))

    return op_types_to_quantize


def configure_ort(
    op_types: List[str], op_types_to_quantize: List[str], trt_extra_plugin_lib_paths=None
):
    """Configure and patches ORT to support ModelOpt ONNX quantization."""
    # Register some new QDQ operators on top of ORT
    QDQRegistry["BatchNormalization"] = QDQNormalization
    QDQRegistry["ConvTranspose"] = QDQConvTranspose
    QDQRegistry["LRN"] = QDQNormalization  # Example: caffenet-12.onnx
    QDQRegistry["HardSwish"] = (
        QDQOperatorBase  # Example: mobilenet_v3_opset17, efficientvit_b3_opset17
    )

    # Patch ORT modules to fix bugs and support some edge cases
    patch_ort_modules(trt_extra_plugin_lib_paths)

    # Remove copy, reduction and activation ops from ORT QDQ registry
    for op_type in [
        "ArgMax",
        "Concat",
        "EmbedLayerNormalization",
        "Gather",
        "InstanceNormalization",
        "LeakyRelu",
        "Pad",
        "Relu",
        "Reshape",
        "Resize",
        "Sigmoid",
        "Softmax",
        "Split",
        "Squeeze",
        "Transpose",
        "Unsqueeze",
        "Where",
    ]:
        if op_type in QLinearOpsRegistry:
            del QLinearOpsRegistry[op_type]
        if op_type in QDQRegistry:
            del QDQRegistry[op_type]

    # Prepare TensorRT friendly quantization settings
    trt_guided_options = {
        "QuantizeBias": False,
        "ActivationSymmetric": True,
        "OpTypesToExcludeOutputQuantization": op_types,  # No output quantization
        "AddQDQPairToWeight": True,  # Instead of quantizing the weights, add QDQ node
        "QDQOpTypePerChannelSupportToAxis": {
            "Conv": 0,
            "ConvTranspose": 1,
        },  # per_channel should be True
        "DedicatedQDQPair": True,
        "ForceQuantizeNoInputCheck": (
            # By default, for some latent operators like MaxPool, Transpose, etc.,
            # ORT does not quantize if their input is not quantized already.
            True
        ),
        "TrtExtraPluginLibraryPaths": trt_extra_plugin_lib_paths,
    }

    quantizable_op_types = get_quantizable_op_types(op_types_to_quantize)
    return trt_guided_options, quantizable_op_types
