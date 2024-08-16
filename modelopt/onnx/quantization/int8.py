# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Perform INT8 quantization of an ONNX model, and returns the ONNX ModelProto."""

import logging
from typing import List, Tuple

import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node
from onnxruntime.quantization import CalibrationMethod
from onnxruntime.quantization.calibrate import CalibrationDataReader

from modelopt.onnx.quantization.calib_utils import import_scales_from_calib_cache
from modelopt.onnx.quantization.graph_utils import (
    build_non_residual_input_map,
    classify_partition_nodes,
    filter_quantizable_kgen_heads,
    find_nodes_to_exclude,
    remove_partial_input_qdq,
)
from modelopt.onnx.quantization.ort_patching import _quantize_static as quantize_static
from modelopt.onnx.quantization.ort_utils import configure_ort
from modelopt.onnx.quantization.partitioning import (
    find_fusible_partitions,
    find_non_quantizable_partitions_from_patterns,
    find_quantizable_nodes,
    get_skiped_output_layers,
)
from modelopt.onnx.quantization.qdq_utils import replace_scale_values
from modelopt.onnx.utils import save_onnx

# Set logging level to info
logging.getLogger().setLevel(logging.INFO)


def _find_nodes_to_quantize(
    graph: Graph,
    quantizable_op_types: List[str],
    verbose: bool,
) -> Tuple[List[Node], List[Tuple[Node, Node, str]]]:
    # Build a map of add nodes to their non-residual inputs, i.e. fusible with Conv group
    logging.info("Building non-residual Add input map ...")
    non_residual_inputs = build_non_residual_input_map(graph)

    logging.info(
        "Searching for hard-coded patterns like MHA, LayerNorm, etc. to avoid quantization."
    )
    non_quantizable_hard_coded_partitions = find_non_quantizable_partitions_from_patterns(graph)

    logging.info("Building KGEN/CASK targeted partitions ...")
    # partitioned_nodes keeps track of nodes that are already part of some partition.
    # Certain nodes of those partitions are quantizable. For example, heads.
    partitioned_nodes = set(sum(non_quantizable_hard_coded_partitions, []))
    cask_fusible_partitions, kgen_partitions = find_fusible_partitions(
        graph,
        partitioned_nodes,
        non_residual_inputs,
    )
    if verbose:
        logging.info(
            "CASK fusible partitions:"
            f" {[[node.name for node in partition] for partition in cask_fusible_partitions]}"
        )
        logging.info(
            "KGEN partitions:"
            f" {[[node.name for node in partition] for partition in kgen_partitions]}"
        )

    logging.info("Classifying the partition nodes ...")
    _, quantizable_partition_nodes, no_quantize_inputs = classify_partition_nodes(
        cask_fusible_partitions,
    )
    quantizable_kgen_heads, no_quantize_kgen_inputs = filter_quantizable_kgen_heads(
        cask_fusible_partitions,
        kgen_partitions,
        quantizable_op_types,
    )

    quantizable_nodes = quantizable_kgen_heads + quantizable_partition_nodes
    paritially_quantizable_nodes = [dst for _, dst, _ in no_quantize_inputs]

    # Quantize all inputs of partially quantizable nodes by ORT
    # but remove QDQ from non-quantizable inputs in the post-processing step
    quantizable_nodes.extend(paritially_quantizable_nodes)

    quantizable_nodes.extend(
        find_quantizable_nodes(graph, quantizable_nodes, partitioned_nodes, quantizable_op_types)
    )

    skip_list = get_skiped_output_layers(graph, paritially_quantizable_nodes)
    quantizable_nodes = [node for node in quantizable_nodes if node.name not in skip_list]

    return quantizable_nodes, no_quantize_inputs + no_quantize_kgen_inputs


def quantize(
    onnx_path: str,
    calibration_method: str = "entropy",
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
    """Applies INT8 quantization to an ONNX file using TensorRT/Myelin friendly heuristics.

    Quantization of ['Add', 'AveragePool', 'BatchNormalization', 'Clip', 'Conv', 'ConvTranspose',
    'Gemm', 'GlobalAveragePool', 'MatMul', 'MaxPool', 'Mul'] op types are supported.
    """
    logging.info("Quantization Mode: int8")

    # Take the onnx graph
    onnx_model = onnx.load(onnx_path, load_external_data=use_external_data_format)
    graph = gs.import_onnx(onnx_model)
    graph.toposort()

    # Change the default configuration of ORT quantization
    op_types_to_quantize = op_types_to_quantize or []
    op_types = set([node.op for node in graph.nodes])
    trt_guided_options, quantizable_op_types = configure_ort(
        list(op_types), op_types_to_quantize, trt_extra_plugin_lib_paths
    )
    logging.info(
        "Quantizable op types in the model:"
        f" {[t for t in quantizable_op_types if t in op_types]}"
    )

    no_quantize_inputs = []
    if not nodes_to_quantize:
        quantizable_nodes, no_quantize_inputs = _find_nodes_to_quantize(
            graph,
            quantizable_op_types,
            verbose,
        )
        nodes_to_quantize = [node.name for node in quantizable_nodes]
    if not nodes_to_quantize:
        logging.info(
            "No node or node type is selected for quantization or model does not have them!"
        )
        return
    elif verbose:
        logging.info(f"Selected nodes: {nodes_to_quantize}")

    # Collect node names to exclude from quantization
    nodes_to_exclude = find_nodes_to_exclude(graph, nodes_to_exclude, op_types_to_exclude)

    logging.info(f"Total number of nodes: {len(graph.nodes)}")
    logging.info(f"Skipped node count: {len(nodes_to_exclude)}")
    if verbose:
        logging.info(f"Skipped nodes: {nodes_to_exclude}")

    # Read the calibration cache and quantize nodes for which activation scale values are cached
    if calibration_cache_path:
        act_scales_dict = import_scales_from_calib_cache(calibration_cache_path)
        logging.info(f"Using calibration cache from {calibration_cache_path}")
        iq_quantized_nodes = []
        quantized_tensors = [
            tensor_name.replace("_scale", "") for tensor_name in act_scales_dict.keys()
        ]
        for node in graph.nodes:
            for node_input in node.inputs:
                if node_input.name in quantized_tensors:
                    iq_quantized_nodes.append(node.name)

        logging.info(
            f"Skipping quantization of nodes: {set(nodes_to_quantize) - set(iq_quantized_nodes)}"
        )
        nodes_to_quantize = list(set(nodes_to_quantize).intersection(iq_quantized_nodes))

    # Use ort api to quantize the onnx model
    quantize_static(
        onnx_path,
        output_path,
        calibration_data_reader,
        op_types_to_quantize=op_types_to_quantize,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
        per_channel=True,
        extra_options=trt_guided_options,
        use_external_data_format=use_external_data_format,
        calibrate_method=(
            CalibrationMethod.Entropy
            if calibration_method == "entropy"
            else CalibrationMethod.MinMax
        ),
    )

    if use_external_data_format:
        intermediate_generated_files.append(output_path + ".data")

    # Post-processing of the onnx model after ort quantization
    onnx_model = onnx.load(output_path)
    graph = gs.import_onnx(onnx_model)
    remove_partial_input_qdq(graph, no_quantize_inputs)
    onnx_model = gs.export_onnx(graph)
    if calibration_cache_path:
        replace_scale_values(onnx_model.graph, act_scales_dict)

    if output_path:
        save_onnx(onnx_model, output_path, use_external_data_format)
        logging.info(f"Quantized onnx model is saved as {output_path}")

    return onnx_model
