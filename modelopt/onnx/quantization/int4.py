# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Perform INT4 WoQ on an ONNX model, and returns the ONNX ModelProto."""

import copy
import logging
import math
import os
import tempfile
import time
import warnings
from typing import List, Tuple, cast

from tqdm import tqdm

try:
    import jax.numpy as np

    has_jax = True
except ImportError as e:
    warnings.warn(
        "Using slower INT4 ONNX quantization using numpy. Install JAX"
        f" (https://jax.readthedocs.io/en/latest/installation.html) for faster quantization: {e}"
    )
    import numpy as np

    has_jax = False

import numpy
import onnx
import onnx.numpy_helper as numpy_helper
import onnx_graphsurgeon as gs
from onnxruntime.quantization.calibrate import CalibrationDataReader

import modelopt.onnx.quantization.qdq_utils as qdq
from modelopt.onnx.quantization.gs_patching import patch_gs_modules
from modelopt.onnx.quantization.ort_utils import create_inference_session
from modelopt.onnx.utils import save_onnx

# Set logging level to info
logging.getLogger().setLevel(logging.INFO)

BLOCK_SIZE = 128
NUM_BITS = 4
INT4_SCALE = 7.0
INT4_MIN = -(2 ** (NUM_BITS - 1))  # -8
INT4_MAX = 2 ** (NUM_BITS - 1) - 1  # 7


def _next_block_size_multiple(x: float, block_size: int) -> float:
    return math.ceil(x / block_size) * block_size


def _pad(w: np.ndarray, block_size: int) -> np.ndarray:
    """Pads `w` to next largest multiple of block_size, on axis 0."""
    if w.shape[0] % block_size == 0:
        return w

    pad_width = _next_block_size_multiple(w.shape[0], block_size) - w.shape[0]
    pads = [(0, 0) for _ in range(len(w.shape))]
    pads[0] = (0, pad_width)
    return np.pad(w, pads, mode="constant", constant_values=0)


def _depad(w: np.ndarray, orig_shape: tuple) -> np.ndarray:
    """Depad axis 0 to original shape."""
    if w.shape == orig_shape:
        return w
    return w[0 : orig_shape[0], ...]


def find_scales(w: np.ndarray, block_size: int, alpha: float = 1.0) -> np.ndarray:
    """Find scale factors for `w` via `s = max(w.block(block_size)) / 7`."""
    w = _pad(w, block_size)
    w = w.T
    w_amax = np.abs(w.reshape(-1, block_size)).max(axis=-1)
    s = (w_amax * alpha) / INT4_SCALE
    s_last_dim = w.shape[-1] // block_size
    s_shape = list(w.shape)
    s_shape[-1] = s_last_dim
    return s.reshape(s_shape).T


def rtn(w: np.ndarray, s: np.ndarray, block_size: int) -> np.ndarray:
    """Quantizes `w` with scale factors `s` via Round-to-Nearest.

    Ties are broken by rounding to the nearest even number.
    """
    w_padded = _pad(w, block_size)
    num_blocks = w_padded.shape[0] // s.shape[0]
    w_padded = (
        np.rint(w_padded / s.repeat(num_blocks, axis=0)).clip(INT4_MIN, INT4_MAX).astype(np.int8)
    )
    return _depad(w_padded, w.shape)


def dq_tensor(w: np.ndarray, s: np.ndarray, block_size: int) -> np.ndarray:
    """Dequantizes `w` with scale factors `s`."""
    w_padded = _pad(w, block_size)
    num_blocks = w_padded.shape[0] // s.shape[0]
    w_padded = w_padded * s.repeat(num_blocks, axis=0)
    return _depad(w_padded, w.shape)


def quantize_rtn(
    onnx_model: onnx.onnx_pb.ModelProto,
    gemm_io_type: onnx.TensorProto.DataType,
    dq_only: bool = False,
) -> onnx.onnx_pb.ModelProto:
    """Quantizes `onnx_model` using the RTN (Round-to-Nearest) algorithm.

    This algorithm computes scale factors by computing s = max(abs(block)) / 8, for each block. The
    quantized weights are computed via Q(w) = round_to_even(w / s), where `round_to_even` denotes
    rounding ties to the nearest even integer (i.e. 1.5, 2.5 both round to 2).

    Always selects the first dimension (0) to block over. This is because we must batch over the Cin
    dimension, and in ONNX, weights are always plugged into the RHS (i.e. y = x @ W).
    """
    graph = gs.import_onnx(onnx_model)
    gemm_nodes = [node for node in graph.nodes if node.op in ["Gemm", "MatMul"]]
    gemm_tensors = {}
    act_tensors = []
    for gemm in gemm_nodes:
        for in_tensor in gemm.inputs:
            if not isinstance(in_tensor, gs.Constant):
                continue
            if len(in_tensor.values.shape) == 1:
                # 1D blocked quantization not supported.
                continue
            gemm_tensors[in_tensor.name] = in_tensor
            act_tensors.append(gemm.inputs[0])

    gemm_weights = {name: tensor.values for name, tensor in gemm_tensors.items()}
    scales = {name: find_scales(w, BLOCK_SIZE) for name, w in gemm_weights.items()}
    logging.info("Computed scales.")

    # Change the scale type to the expected type, fp16 by default
    for name, w in scales.items():
        s = scales[name]
        scales[name] = s.astype(onnx.mapping.TENSOR_TYPE_MAP[gemm_io_type].np_dtype)

    # Change the input activation type to the expected type, fp16 by default
    for act_tensor in act_tensors:
        _change_input_type(onnx_model.graph, act_tensor.name, gemm_io_type)

    # Import the update graph
    graph = gs.import_onnx(onnx_model)

    if dq_only:
        # Calculate actual quantized weights.
        gemm_weights_quantized = {}
        for name, w in gemm_weights.items():
            gemm_weights_quantized[name] = rtn(w, scales[name], BLOCK_SIZE)

        qdq.insert_dq_nodes(graph, scales, quantized_weights=gemm_weights_quantized)
    else:
        qdq.insert_qdq_nodes(graph, scales, weight_map=gemm_tensors)

    logging.info(f"Inserted {'DQ' if dq_only else 'Q/DQ'} nodes.")
    return gs.export_onnx(graph)


def quant_tensor(w: np.ndarray, block_size: int, alpha: float = 1.0):
    """Quantize a tensor using alpha etc. and return the quantized tensor."""
    scale = find_scales(w, block_size, alpha)
    wq = rtn(w, scale, block_size)
    return wq, scale


class AWQClipHelper:
    """AWQ calibration helper class."""

    min_alpha = 0.5
    alpha_step = 0.05
    alphas = [round(float(k), 2) for k in np.arange(min_alpha, 1.0, alpha_step)] + [1.0]

    def __init__(self, w, block_size: int):
        """Initializes AWQClipHelper with a module weight."""
        ci, co = w.shape
        self.block_size = block_size if block_size != -1 else w.shape[0]
        w = _pad(w, block_size).T
        self.w_amax = np.abs(w.reshape(-1, block_size)).max(axis=-1)

        self.loss = {
            k: np.zeros((co, math.ceil(ci / self.block_size)), dtype=np.float32)
            for k in AWQClipHelper.alphas
        }
        self.best_loss = np.full_like(self.w_amax, float("inf"))
        self.best_alpha = np.ones_like(self.w_amax)

    def update_best_params(self):
        """Updates the loss dictionary."""
        for alpha, loss in self.loss.items():
            loss = loss.reshape(self.w_amax.shape)
            indices = loss < self.best_loss
            self.best_loss = np.where(indices, loss, self.best_loss)
            self.best_alpha = np.where(indices, alpha, self.best_alpha)


def _clip_search(
    x: np.ndarray,
    w: np.ndarray,
    awq_clip: AWQClipHelper,
    co_bsz: int = 1024,
    max_tokens: int = 64,
):
    """Apply AWQ algorithm on a weight and return optimum alpha.

    This algorithm defines a simple search space for the optimal scales: S = Sx ^ α.
    S is only related to the magnitude of activation Sx, and a single hyper-parameter α is used to balance
    between the protection of salient and non-salient channels. The algorithm finds the best α by a fast grid search
    over the interval of [0, 1] (0 means do not scale; 1 corresponds to the most aggressive scaling).
    Further weight clipping is also applied by minimizing the MSE error.
    """
    # Select max_tokens from input
    x = np.reshape(x, (-1, x.shape[-1]))  # _, ci
    x = x[0 :: max(1, x.shape[0] // max_tokens)]  # max_tokens, ci

    ci, co = w.shape
    block_size = awq_clip.block_size

    # Pad weight and input if necessary
    if ci % block_size != 0:
        w = _pad(w, block_size)
        x = _pad(x.T, block_size).T

    # Make a copy of the original padded weight to quantize with generated scales
    w_copy = copy.deepcopy(w)

    # Reshape weight and input for batch processing over co dimension
    w = w.T  # co, ci
    w = w.reshape(co, 1, -1, block_size)  # co, 1, n_block, block_size
    x = x.reshape(1, x.shape[0], -1, block_size)  # 1, max_tokens, n_block, block_size

    # Loop over co dimension of the weight and generate scales
    for co_batch in range(math.ceil(co / co_bsz)):
        slice_s, slice_e = co_batch * co_bsz, min((co_batch + 1) * co_bsz, co)
        weight = w[slice_s:slice_e]
        org_out = np.sum(x * weight, axis=-1)  # co_bsz, max_tokens, n_block

        # Compute loss for each alpha value
        for alpha in awq_clip.loss.keys():
            # Perform QDQ on the whole original weight tensor
            qw, scales = quant_tensor(w_copy, block_size, alpha)
            cur_w = dq_tensor(qw, scales, block_size)

            # Reshape before getting the batch of size co_bsz to multiply with input
            cur_w = cur_w.T  # ci, co -> co, ci
            cur_w = cur_w.reshape(co, 1, -1, block_size)  # co, 1, n_block, block_size
            cur_w = cur_w[slice_s:slice_e]

            # Compute loss for each batch
            cur_out = np.sum(x * cur_w, axis=-1)  # co_bsz, max_tokens, n_block
            loss = np.mean(np.power((org_out - cur_out), 2), axis=1)  # co_bsz, n_block
            if has_jax:
                awq_clip.loss[alpha] += awq_clip.loss[alpha].at[slice_s:slice_e].set(loss)
            else:
                awq_clip.loss[alpha][slice_s:slice_e] += loss

    # Update the best alpha value for the weight blocks
    awq_clip.update_best_params()


def _find_quantizable_weights(
    graph: onnx.onnx_pb.GraphProto,
) -> List[Tuple[onnx.onnx_pb.ValueInfoProto, onnx.onnx_pb.ValueInfoProto, bool, int]]:
    """Finds the quantizable weights from the graph."""
    wa_pack = []
    gemm_nodes = [node for node in graph.node if node.op_type in ["Gemm", "MatMul"]]
    initializer_idxs = {initializer.name: idx for idx, initializer in enumerate(graph.initializer)}
    for gemm in gemm_nodes:
        if gemm.input[0] in initializer_idxs:
            # Ex. two const input to MatMul_115 in fastvit0.onnx
            # Note. RTN algorithm will quantize these weights though
            continue

        if gemm.input[1] not in initializer_idxs:
            continue

        weight_tensor = graph.initializer[initializer_idxs[gemm.input[1]]]
        if len(weight_tensor.dims) == 1:  # 1D blocked quantization not supported
            continue

        gemm_io_type = cast(int, weight_tensor.data_type)

        act_tensor = onnx.helper.ValueInfoProto()
        act_tensor.name = gemm.input[0]

        # TODO: support transA by transposing activation tensors in _clip_search
        do_transpose = gemm.op_type == "Gemm" and any(
            [attr.name == "transB" and attr.i > 0 for attr in gemm.attribute]
        )

        wa_pack.append((act_tensor, weight_tensor, do_transpose, gemm_io_type))

    return wa_pack


def _augment_graph(
    graph: onnx.onnx_pb.GraphProto,
    wa_pack: List[Tuple[gs.Tensor, gs.Tensor, bool, int]],
):
    """Extend graph outputs with MatMuls activation input."""
    augmented_outputs = set([tensor.name for tensor in graph.output])
    for act_tensor, _, _, _ in wa_pack:
        if act_tensor.name not in augmented_outputs:
            graph.output.append(act_tensor)
            augmented_outputs.add(act_tensor.name)


def _change_input_type(
    graph: onnx.onnx_pb.GraphProto, input_name: str, gemm_io_type: onnx.TensorProto.DataType
):
    # Find the corresponding value info in the graph
    done = False
    for value_info in graph.value_info:
        if value_info.name == input_name:
            value_info.type.tensor_type.elem_type = gemm_io_type
            done = True
            break

    if not done:
        # If input not in value_info, it must be a graph input
        for input_info in graph.input:
            if input_info.name == input_name:
                input_info.type.tensor_type.elem_type = gemm_io_type
                break


def quantize_awq_clip(
    onnx_model: onnx.onnx_pb.ModelProto,
    data_reader: CalibrationDataReader,
    use_external_data_format: bool,
    force_fp16: bool = False,
) -> onnx.onnx_pb.ModelProto:
    """Quantizes `onnx_model` using the Activation aware quantization a.k.a AWQ algorithm."""
    logging.info("Finding quantizable weights and augmenting graph output with input activations")
    t = time.time()
    augmented_model = copy.deepcopy(onnx_model)
    graph = augmented_model.graph

    # Collect quantizable weight tensors
    wa_pack = _find_quantizable_weights(graph)

    # Add input activations to graph output
    _augment_graph(augmented_model.graph, wa_pack)
    logging.info(f"Augmenting took {time.time() - t} seconds")

    scales = {}
    gemm_weights_quantized = {}

    t = time.time()

    # Create a temp file for augmented model
    augmented_onnx_file, augmented_onnx_path = tempfile.mkstemp(suffix=".onnx")
    os.close(augmented_onnx_file)

    # TODO: ONNX version issue, onnx_export uses current ONNX IR version.
    augmented_model.ir_version = 9
    save_onnx(augmented_model, augmented_onnx_path, use_external_data_format)
    logging.info(f"Saving the model took {time.time() - t} seconds")

    # Creating inference session and preparing inputs for calibration
    session = create_inference_session(augmented_onnx_path)
    inputs = []
    for inp_d in data_reader:
        inputs.append(inp_d)
        assert isinstance(inp_d, dict)

    # Apply AWQ clip on selected weights
    t = time.time()
    alphas = {}
    for i in tqdm(range(len(wa_pack)), desc="Running clip search..."):

        act_tensor, weight_tensor, do_transpose, gemm_io_type = wa_pack[i]

        # First capture all the  activation values after calibration data sweep
        output_dicts = {}
        for inp_d in inputs:
            np_inp_d = {name: numpy.asarray(tensor) for name, tensor in inp_d.items()}
            output = session.run([act_tensor.name], np_inp_d)
            output_dicts.setdefault(act_tensor.name, []).append(output[0])

        # Concatenating the activation tensors over all calib data
        x = np.concatenate(output_dicts[act_tensor.name], axis=0)  # n_token, ci
        w = numpy_helper.to_array(
            weight_tensor, base_dir=os.path.dirname(augmented_onnx_path)
        ).copy()
        if do_transpose:
            w = w.T

        awq_clip = AWQClipHelper(w, BLOCK_SIZE)
        _clip_search(x, w, awq_clip)
        alphas[weight_tensor.name] = awq_clip.best_alpha

    logging.info(f"Clip search for all weights took {time.time() - t} seconds")

    # Compute quantized weights and scales which are needed for DQ nodes
    t = time.time()
    for i in tqdm(range(len(wa_pack)), desc="Quantizing the weights..."):

        act_tensor, weight_tensor, do_transpose, gemm_io_type = wa_pack[i]
        gemm_io_type = cast(onnx.TensorProto.DataType, gemm_io_type)

        if force_fp16:
            gemm_io_type = onnx.TensorProto.FLOAT16

        w = numpy_helper.to_array(
            weight_tensor, base_dir=os.path.dirname(augmented_onnx_path)
        ).copy()
        if do_transpose:
            w = w.T

        alpha = alphas.get(weight_tensor.name, 1)
        qw, scale = quant_tensor(w, BLOCK_SIZE, alpha)
        if do_transpose:
            qw = qw.T
            scale = scale.T
        scales[weight_tensor.name] = scale.astype(
            onnx.mapping.TENSOR_TYPE_MAP[gemm_io_type].np_dtype
        )
        gemm_weights_quantized[weight_tensor.name] = numpy.asarray(qw).astype(numpy.int8)

        # Change the input activation type to the expected type, fp16 by default
        # TODO: cast input C for Gemm
        _change_input_type(onnx_model.graph, act_tensor.name, gemm_io_type)
    logging.info(f"Quantizing actual weights took {time.time() - t} seconds")

    logging.info("Inserting DQ nodes using quantized weights and scales ...")
    t = time.time()
    graph_gs = gs.import_onnx(onnx_model)
    qdq.insert_dq_nodes(graph_gs, scales, quantized_weights=gemm_weights_quantized)
    logging.info(f"Inserting DQ nodes took {time.time() - t} seconds")

    logging.info("Exporting the quantized graph ...")
    t = time.time()
    model = gs.export_onnx(graph_gs)
    model.ir_version = 9
    logging.info(f"Exporting took {time.time() - t} seconds")

    try:
        os.remove(augmented_onnx_path)
        if use_external_data_format:
            os.remove(augmented_onnx_path + "_data")
    except OSError:
        logging.warn("Augmented ONNX model or external data file was not found!")

    return model


def quantize(
    onnx_path: str,
    calibration_method: str = "awq_clip",
    calibration_data_reader: CalibrationDataReader = None,
    use_external_data_format: bool = True,
    output_path: str = None,
) -> onnx.onnx_pb.ModelProto:
    """Applies INT4 WoQ (Weight-Only-Quantization) to an ONNX file.

    Currently only GEMM quantization is supported.
    """
    logging.info("Quantization Mode: int4")
    gemm_io_type: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT

    # Patch GS modules to support INT4.
    patch_gs_modules()

    if "trt" in calibration_method:
        qdq.use_trt_qdq_ops()

    onnx_model = onnx.load(onnx_path, load_external_data=use_external_data_format)

    if calibration_method in ["rtn", "rtn_dq", "rtn_trt", "rtn_trt_dq"]:
        onnx_model = quantize_rtn(onnx_model, gemm_io_type, dq_only="dq" in calibration_method)
    elif calibration_method in ["awq_clip", "awq_clip_trt"]:
        onnx_model = quantize_awq_clip(
            onnx_model, calibration_data_reader, use_external_data_format
        )
    else:
        raise RuntimeError(f"Unsupported calibration method: '{calibration_method}'")

    if output_path:
        save_onnx(onnx_model, output_path, use_external_data_format)
        logging.info(f"Quantized onnx model is saved as {output_path}")

    return onnx_model
