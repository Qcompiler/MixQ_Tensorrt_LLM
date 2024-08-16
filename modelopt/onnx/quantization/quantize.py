# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Convert ONNX model without QDQ nodes + calib data into ONNX model with QDQ nodes.

Typically quantizing linear operations like Conv, MatMul etc. gives most of the performance boost.
But there are many other ops that are quantizable (aka low precision kernels available) and provides
optimal performance with lower accuracy drop. The default op types that this ONNX ptq tool quantizes
in different quantization modes are: INT8: ['Add', 'AveragePool', 'BatchNormalization', 'Clip',
'Conv', 'ConvTranspose', 'Gemm', 'GlobalAveragePool', 'MatMul', 'MaxPool', 'Mul'], INT4: ['Gemm',
'MatMul'], FP8: ['Conv', 'Gemm', 'MatMul']. The tool inserts QDQ nodes following compiler friendly
patterns and generates an explicit ONNX model.
"""
import logging
import os
import shutil
import tempfile
from typing import List, Tuple

import numpy as np
import onnx
import onnx.onnx_cpp2py_export.checker as C  # noqa: N812
import onnx_graphsurgeon as gs
from onnxmltools.utils.float16_converter import convert_float_to_float16

from modelopt.onnx.quantization.calib_utils import (
    CalibrationDataProvider,
    CalibrationDataType,
    RandomDataProvider,
)
from modelopt.onnx.quantization.fp8 import quantize as quantize_fp8
from modelopt.onnx.quantization.graph_utils import (
    add_fp16_fp32_cast,
    find_mha_partitions,
    insert_fp8_mha_casts,
    print_stat,
)
from modelopt.onnx.quantization.int4 import quantize as quantize_int4
from modelopt.onnx.quantization.int8 import quantize as quantize_int8
from modelopt.onnx.utils import (
    duplicate_shared_constants,
    name_onnx_nodes,
    save_onnx,
)

__all__ = ["quantize"]

QUANT_MODES = [
    "int8",  # INT8 quantization of both scales and activations.
    "int4_rtn",  # INT4 weight-only quantization. Inserts Q and DQ nodes around each eligible weight tensor.
    "int4_rtn_dq",  # INT4 weight-only quantization. Directly computes the INT4 weights, and only inserts DQ nodes.
    "int4_rtn_trt",  # Same as `int4_rtn`, but exports TRT custom Q/DQ nodes instead.
    "int4_rtn_trt_dq",  # Same as `int4_rtn_dq`, but exports TRT custom DQ nodes instead.
    "int4_awq_clip",  # INT4 AWQ Clip. Inserts DQ nodes for each eligible weight tensor.
    "int4_awq_clip_trt",  # Same as `int4_awq_clip`, but exports TRT custom DQ nodes instead.
    "fp8",
]

# Set logging level to info
logging.getLogger().setLevel(logging.INFO)


def _preprocess_onnx(
    onnx_path: str,
    use_external_data_format: bool,
    output_path: str,
    trt_plugins_precision: List[str],
) -> Tuple[str, List[str], bool]:
    # Load the model and weights
    onnx_model = onnx.load(onnx_path, load_external_data=use_external_data_format)

    # Check if there's a custom TensorRT op in the ONNX model
    has_custom_op = np.any([node.domain == "trt.plugins" for node in onnx_model.graph.node])

    # Per-Channel support with QDQ format requires onnx opset version 13 or above
    ai_onnx_domain = [
        opset
        for opset in onnx_model.opset_import
        if not opset.domain or opset.domain in ["ai.onnx", "ai.onnx.contrib"]
    ]
    opset_version = ai_onnx_domain[0].version
    logging.info(f"Model {onnx_path} with opset_version {opset_version} is loaded.")

    intermediate_generated_files = []
    output_dir = os.path.dirname(output_path)
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]

    required_opset_version = 13
    if opset_version < required_opset_version and opset_version != 1:
        opset_version = required_opset_version
        onnx_model = onnx.version_converter.convert_version(onnx_model, opset_version)
        onnx_path = os.path.join(output_dir, f"{model_name}_opset{opset_version}.onnx")
        save_onnx(onnx_model, onnx_path, use_external_data_format)
        logging.info(f"Model is cloned to {onnx_path} with opset_version {opset_version}.")
        intermediate_generated_files.append(onnx_path)

    # Sometimes input onnx model does not contain the node names
    # This tool depends on those names, so we assign names if needed
    graph = onnx_model.graph
    is_named = name_onnx_nodes(graph)
    onnx_model, is_duplicated_constant = duplicate_shared_constants(onnx_model)  # FasterViT-0, eef

    if is_named or is_duplicated_constant:
        onnx_path = os.path.join(output_dir, f"{model_name}_named.onnx")
        save_onnx(onnx_model, onnx_path, use_external_data_format)
        logging.info(f"Model is cloned to {onnx_path} after naming the nodes.")
        intermediate_generated_files.append(onnx_path)

    # If custom op precisions are given, check if they're fp16. If so, add cast_to_fp16 before all inputs and
    # cast_to_fp32 after all outputs.
    if trt_plugins_precision:
        custom_ops_to_cast = []
        for trt_plugin_precision in trt_plugins_precision:
            assert ":" in trt_plugin_precision, (
                "Plugin precision is incorrectly formatted."
                " Please check that it's in the format <op_type>:<precision>."
            )
            op_type, precision = trt_plugin_precision.split(":")
            if precision == "fp16":
                custom_ops_to_cast.append(op_type)
        if custom_ops_to_cast:
            onnx_path = add_fp16_fp32_cast(onnx_path, custom_ops_to_cast)
            logging.info("Adding cast nodes related to custom ops to match requested precisions.")
            intermediate_generated_files.append(onnx_path)
    return onnx_path, intermediate_generated_files, has_custom_op


def quantize(
    onnx_path: str,
    calibration_data: CalibrationDataType = None,
    calibration_method: str = None,
    calibration_cache_path: str = None,
    op_types_to_quantize: List[str] = None,
    op_types_to_exclude: List[str] = None,
    nodes_to_quantize: List[str] = None,
    nodes_to_exclude: List[str] = None,
    use_external_data_format: bool = False,
    keep_intermediate_files: bool = False,
    output_path: str = None,
    verbose: bool = False,
    quantize_mode: str = "int8",
    trt_plugins: str = None,
    trt_plugins_precision: List[str] = None,
    high_precision_dtype: str = "fp16",
    mha_accumulation_dtype: str = "fp32",
    disable_mha_qdq: bool = False,
) -> None:
    """Quantize the given onnx model.

    Args:
        onnx_path:
            Path to the input onnx model.
        calibration_data:
            Calibration data, either a numpy array or list/dict of numpy array.
        calibration_method:
            Calibration method choices for int8, options={entropy (default), minmax}.
        calibration_cache_path:
            Pre-calculated activation tensor ranges aka calibration cache path.
        op_types_to_quantize:
            List of types of operators to quantize. When this list is not None, only the types in this list
            are quantized. Example: ['Conv'] indicates that only ops of type 'Conv' should be quantized.
            If this list is None (default), all supported operators are quantized.
            This flag does not support regular expression.
        op_types_to_exclude:
            List of types of operators to exclude from quantization.
            This flag does not support regular expression.
        nodes_to_quantize:
            List of node names to quantize. When this list is not None, only the nodes in this list
            are quantized. Example: ['Conv__224', 'Conv__252'].
            If this list is None (default), all supported nodes are quantized.
            This flag does not support regular expression.
        nodes_to_exclude:
            List of nodes names to exclude. The nodes in this list will be excluded from quantization
            when it is not None. This flag supports regular expression.
        use_external_data_format:
            If not None, this path will be used to store the weights of the quantized model.
        keep_intermediate_files:
            If False, only save the converted ONNX files for the user. Otherwise, keep all intermediate files
             generated during the ONNX models' conversion/calibration.
        output_path:
            Output filename to save the converted ONNX model.
            If None, save in the same directory as the original ONNX model with .quant suffix.
        verbose:
            Prints details of node partition, selection etc. throughout the quantization process.
        quantize_mode:
            Quantization mode. One of ['int8', 'int4_rtn', 'int4_rtn_dq', 'int4_rtn_trt', 'int4_rtn_trt_dq',
            'int4_awq_clip', 'int4_awq_clip_trt', 'fp8']. 'int8' by default. Any INT4-based mode is Gemm, MatMul
            weight-only and FP8 mode is Conv, Gemm and MatMul only quantization.
        trt_plugins:
            Specifies custom TensorRT plugin library paths in .so format (compiled shared library).
            For multiple paths, separate them with a semicolon, i.e.: "lib_1.so;lib_2.so".
            If this is not None, the TensorRTExecutionProvider is invoked, meaning that TensorRT is a requirement.
        trt_plugins_precision:
            A space-separated list indicating the precision for each custom op.
            Each item should have the format <op_type>:<precision>, where precision can be fp32 (default) or fp16.
            For example: op_type_1:fp16 op_type_2:fp32.
        high_precision_dtype:
            High precision dtype. One of ['fp32', 'fp16']. 'fp16' by default.
            If quantize_mode == 'fp8' and high_precision_dtype == 'fp16', model's weight and
            activation will be converted to fp16.
        mha_accumulation_dtype:
            MHA accumulation dtype. One of ['fp32', 'fp16']. 'fp32' by default.
            If quantize_mode == 'fp8' and high_precision_dtype == 'fp32', Cast nodes will be added to
            MHA's bmm1 and bmm2's input and output tensors.
        disable_mha_qdq:
            Don't add Q/DQ layers to MatMuls in MHA pattern.

    Returns:
        None, write the quantized onnx model in the same directory with filename like "<model_name>.quant.onnx".
    """
    # quantize_static creates a shape-inferred copy at the input model's directory
    # Needs to check if we have write permission to this directory
    assert onnx_path.endswith(".onnx") or onnx_path.endswith(".pb")
    if not os.access(os.path.dirname(os.path.abspath(onnx_path)), os.W_OK):
        old_dir = os.path.dirname(os.path.abspath(onnx_path))
        tmp_dir = tempfile.mkdtemp()
        logging.info(f"Directory {old_dir} is not writable, store intermediate files in {tmp_dir}")
        onnx_path = os.path.join(tmp_dir, os.path.basename(onnx_path))

        # We assume that the model directory contains only model related weights and protobuf file
        # Anything extra in the model directory will be copied unnecessarily
        for file in os.listdir(old_dir):
            old_file_path = os.path.join(old_dir, file)
            new_file_path = os.path.join(tmp_dir, file)
            if os.path.isfile(old_file_path):
                shutil.copy(old_file_path, new_file_path)

    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    if not output_path:
        output_dir = os.path.dirname(onnx_path)
        output_path = os.path.join(output_dir, f"{model_name}.quant.onnx")
        logging.info(f"No output path specified, save quantized model to {output_path}")

    # We need to preprocess the model with naming, weight duplication etc.
    onnx_path, intermediate_generated_files, has_custom_op = _preprocess_onnx(
        onnx_path, use_external_data_format, output_path, trt_plugins_precision
    )
    # If the model has a custom op and no plugin path was given, assume that this custom op is being implemented
    # by a TRT native plugin. In order to enable the TRT EP, 'trt_extra_plugin_lib_paths' needs to be != None.
    if has_custom_op and not trt_plugins:
        trt_plugins = ""

    # Use random scales if calibration data is not supplied
    if calibration_data is None:
        calibration_data_reader = RandomDataProvider(onnx_path)
    else:
        calibration_data_reader = CalibrationDataProvider(onnx_path, calibration_data)

    # Don't add Q/DQ layers to MatMuls in MHA pattern if disable_mha_qdq is set.
    if disable_mha_qdq:
        onnx_model = onnx.load(onnx_path, load_external_data=use_external_data_format)
        graph = gs.import_onnx(onnx_model)
        mha_partitions = find_mha_partitions(graph)
        for mha_partition in mha_partitions:
            nodes_to_exclude.append(mha_partition[0].name)
            nodes_to_exclude.append(mha_partition[2].name)

    final_path = output_path
    if quantize_mode == "fp8" and high_precision_dtype == "fp16":
        output_path = os.path.join(
            os.path.dirname(output_path),
            f"{os.path.splitext(os.path.basename(output_path))[0]}.fp32+fp8.onnx",
        )
        intermediate_generated_files.append(output_path)

    if quantize_mode in ["fp8", "int8"]:
        quantize_func = quantize_int8 if quantize_mode == "int8" else quantize_fp8
        default_calibration_method = "entropy" if quantize_mode == "int8" else "distribution"
        onnx_model = quantize_func(
            onnx_path=onnx_path,
            calibration_method=calibration_method or default_calibration_method,
            calibration_data_reader=calibration_data_reader,
            calibration_cache_path=calibration_cache_path,
            op_types_to_quantize=op_types_to_quantize,
            op_types_to_exclude=op_types_to_exclude,
            nodes_to_quantize=nodes_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            use_external_data_format=use_external_data_format,
            intermediate_generated_files=intermediate_generated_files,
            output_path=output_path,
            verbose=verbose,
            trt_extra_plugin_lib_paths=trt_plugins,
        )
    elif "int4" in quantize_mode:
        onnx_model = quantize_int4(
            onnx_path=onnx_path,
            calibration_method=calibration_method or "awq_clip",
            calibration_data_reader=calibration_data_reader,
            use_external_data_format=use_external_data_format,
            output_path=output_path,
        )
    else:
        logging.error(f"Invalid quantization mode choice: {quantize_mode}")

    if quantize_mode == "fp8" and high_precision_dtype == "fp16":
        # We need to convert float to float16 so as to speed up layers like LayerNorm or GroupNorm.
        logging.info("Converting float tensors to float16")
        onnx_model = convert_float_to_float16(
            onnx_model, keep_io_types=True, disable_shape_infer=True
        )

        if mha_accumulation_dtype == "fp32":
            # Insert Cast nodes in MHA's BMM1 and BMM2's input and output tensors because
            # Myelin only has FP32 accumulation kernels for FP8 MHAs.
            logging.info("Inserting Cast nodes to enable FP8+FP16 MHA")
            onnx_model = insert_fp8_mha_casts(onnx_model)

        logging.info(f"Save fp8+fp16 model to {final_path}")
        onnx.save_model(onnx_model, final_path, save_as_external_data=True)
        output_path = final_path

    # Collect and print stats of the quantized model
    if onnx_model:
        print_stat(gs.import_onnx(onnx_model), verbose)

    # Check if intermediate files should be deleted
    if not keep_intermediate_files:
        for file in intermediate_generated_files:
            os.remove(file)

    # Check if the quantized model is valid
    try:
        onnx.checker.check_model(output_path)
    except C.ValidationError as e:
        logging.warn("ONNX model checker failed, check your deployment status.")
        logging.warn(e)
