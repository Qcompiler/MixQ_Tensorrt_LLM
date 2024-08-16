# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Perform FP8 GEMM only quantization of an ONNX model, and returns the ONNX ModelProto."""

import logging
from typing import List

import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.graph import Graph
from onnxruntime.quantization import (
    CalibrationMethod,
    quantize_static,
)
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.quant_utils import (
    QuantType,
)

from modelopt.onnx.quantization.graph_utils import find_nodes_to_exclude
from modelopt.onnx.quantization.ort_utils import configure_ort


def _find_unpaddable_fp8_convs_to_exclude(graph: Graph):
    # In CASK 5.8, the input and output channel alignment requirement for FP8
    # conv kernels for input and output type FP8E4M3 are both 16.
    # Check myelin/src/compiler/optimizer/cask_impl.cpp::collect_conv_constraints
    # for detail.
    unpaddable_conv_nodes = []
    for node in graph.nodes:
        if node.op == "Conv":
            weight = node.inputs[1]
            output_channel = weight.shape[0]
            input_channel = weight.shape[1]
            if output_channel % 16 != input_channel % 16:
                logging.info(f"Found unpaddable conv for FP8: {node.name}")
                unpaddable_conv_nodes.append(node.name)

    return unpaddable_conv_nodes


def quantize(
    onnx_path: str,
    calibration_method: str = "distribution",
    calibration_data_reader: CalibrationDataReader = None,
    calibration_cache_path: str = None,
    op_types_to_quantize: List[str] = None,
    op_types_to_exclude: List[str] = None,
    nodes_to_quantize: List[str] = None,
    nodes_to_exclude: List[str] = None,
    use_external_data_format: bool = True,
    intermediate_generated_files: List[str] = [],
    output_path: str = None,
    verbose: bool = False,
    trt_extra_plugin_lib_paths: str = None,
) -> onnx.onnx_pb.ModelProto:
    """Applies FP8 GEMM only quantization to an ONNX file.

    Currently ['Conv', 'Gemm', 'MatMul'] quantization is supported.
    """
    logging.info("Quantization Mode: fp8")

    # Take the onnx graph
    onnx_model = onnx.load(onnx_path, load_external_data=use_external_data_format)
    graph = gs.import_onnx(onnx_model)
    graph.toposort()

    # Quantizable op type is limited to Conv, Gemm and Matmul for fp8
    fp8_supported_op_types = ["Gemm", "MatMul", "Conv"]
    op_types_to_quantize = op_types_to_quantize or fp8_supported_op_types
    if not set(op_types_to_quantize) <= set(fp8_supported_op_types):
        raise RuntimeError(
            f"Unsupported op types in fp8 mode: '{set(op_types_to_quantize) - set(fp8_supported_op_types)}'"
        )

    # Change the default configuration of ORT quantization
    op_types = set([node.op for node in graph.nodes])
    trt_guided_options, quantizable_op_types = configure_ort(list(op_types), op_types_to_quantize)
    logging.info(
        "Quantizable op types in the model:"
        f" {[t for t in quantizable_op_types if t in op_types]}"
    )

    nodes_to_quantize = [node.name for node in graph.nodes if node.op in op_types_to_quantize]
    if not nodes_to_quantize:
        logging.info(
            "No node or node type is selected for quantization or model does not have them!"
        )
        return

    # Collect node names to exclude from quantization
    nodes_to_exclude = find_nodes_to_exclude(graph, nodes_to_exclude, op_types_to_exclude)
    nodes_to_exclude.extend(_find_unpaddable_fp8_convs_to_exclude(graph))

    logging.info(f"Total number of nodes: {len(graph.nodes)}")
    logging.info(f"Skipped node count: {len(nodes_to_exclude)}")
    if verbose:
        logging.info(f"Skipped nodes: {nodes_to_exclude}")

    quantize_static(
        onnx_path,
        output_path,
        calibration_data_reader,
        op_types_to_quantize=op_types_to_quantize,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
        # Enabling this causes a model with both FP8 and INT8 nodes which TRT is not happy with
        per_channel=False,
        use_external_data_format=use_external_data_format,
        # This is the only available calibrate method for fp8 now
        calibrate_method=CalibrationMethod.Distribution,
        extra_options=trt_guided_options,
        activation_type=QuantType.QFLOAT8E4M3FN,
        weight_type=QuantType.QFLOAT8E4M3FN,
    )

    logging.info(f"Quantized onnx model is saved as {output_path}")
    return onnx.load(output_path)
