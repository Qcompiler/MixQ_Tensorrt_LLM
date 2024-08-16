# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Various utils to support inserting Q/DQ nodes."""

from typing import Dict, Sequence, Set

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import numpy_helper

QUANTIZE_NODE_NAME = "QuantizeLinear"
DEQUANTIZE_NODE_NAME = "DequantizeLinear"


def use_trt_qdq_ops():
    """Globally set node names to TRT custom names."""
    global QUANTIZE_NODE_NAME
    QUANTIZE_NODE_NAME = "TRT_INT4QuantizeLinear"
    global DEQUANTIZE_NODE_NAME
    DEQUANTIZE_NODE_NAME = "TRT_INT4DequantizeLinear"


def _wq_name(name: str):
    return name + "_i4"


def _scale_name(name: str):
    return name + "_scale"


def _zp_name(name: str):
    return name + "_zp"


def _q_name(name: str):
    return name + "_QuantizeLinear"


def _q_out_name(name: str):
    return name + "_QuantizeLinear_Output"


def _dq_name(name: str):
    return name + "_DequantizeLinear"


def _dq_out_name(name: str):
    return name + "_DequantizeLinear_Output"


def make_gs_quantized_weight(name: str, wq: np.ndarray, dtype) -> gs.Constant:
    """Create a GraphSurgeon tensor from a quantized weight tensor.

    `name` is the desired _basename_ of the tensor.
    """
    return gs.make_constant(_wq_name(name), np.asarray(wq), dtype)


def make_gs_zp(name: str, shape: Sequence[int], dtype) -> gs.Constant:
    """Create a GraphSurgeon zero-point tensor of all zeroes with the given shape.

    `name` is the desired _basename_ of the tensor.
    """
    return gs.make_constant(
        _zp_name(name),
        np.zeros(shape, dtype=onnx.mapping.TENSOR_TYPE_MAP[int(dtype)].np_dtype),
        dtype,
    )


def make_gs_scale(name: str, scale: np.ndarray) -> gs.Constant:
    """Create a GraphSurgeon scale tensor from the given numpy array.

    `name` is the desired _basename_ of the tensor.
    """
    return gs.Constant(_scale_name(name), np.asarray(scale))


def make_gs_quantize_output(
    name: str, shape: Sequence[int], dtype: onnx.TensorProto.DataType
) -> gs.Variable:
    """Create a GraphSurgeon variable representing the output of a quantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.make_variable(_q_out_name(name), dtype=dtype, shape=shape)


def make_gs_quantize_node(
    name: str, inputs: Sequence[gs.Tensor], outputs: Sequence[gs.Tensor]
) -> gs.Node:
    """Create a GraphSurgeon Quantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Node(
        QUANTIZE_NODE_NAME,
        name=_q_name(name),
        inputs=inputs,
        outputs=outputs,
    )


def make_gs_dequantize_output(
    name: str,
    shape: Sequence[int],
    dtype: np.dtype,
) -> gs.Variable:
    """Create a GraphSurgeon variable representing the output of a quantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Variable(_dq_out_name(name), dtype=dtype, shape=shape)


def make_gs_dequantize_node(
    name: str, inputs: Sequence[gs.Tensor], outputs: Sequence[gs.Tensor]
) -> gs.Node:
    """Create a GraphSurgeon Dequantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Node(
        DEQUANTIZE_NODE_NAME,
        name=_dq_name(name),
        inputs=inputs,
        outputs=outputs,
    )


def _postprocess_qdq(
    graph: gs.Graph,
    orig_weight_names: Set[str],
    q_nodes: Dict[str, gs.Node] = {},
    dq_nodes: Dict[str, gs.Node] = {},
):
    # Inserts all newly created nodes to graph.
    # Update all consumers of original initializers to point to the DQ nodes.
    for node in graph.nodes:
        for i in range(len(node.inputs)):
            key = node.inputs[i].name
            if key not in orig_weight_names:
                continue
            node.inputs[i] = dq_nodes[key].outputs[0]

    # Insert new nodes.
    graph.nodes.extend(q_nodes.values())
    graph.nodes.extend(dq_nodes.values())

    graph.cleanup()
    graph.toposort()


def insert_dq_nodes(
    graph: gs.Graph, scales: Dict[str, np.ndarray], quantized_weights: Dict[str, np.ndarray]
):
    """Insert new initializers and DQ nodes into graph.

    Args:
        graph: The graph to modify.
        weights: A map from ONNX initializer name to tensor.
        scales: A map from ONNX initializer name to desired scale factor for that initializer.
        dq_only: Whether to only insert dq nodes.
    """

    def _insert_helper(
        name: str,
        wq: np.ndarray,
        scale: np.ndarray,
        dq_nodes: Dict[str, gs.Node],
    ):
        wq_tensor = make_gs_quantized_weight(name, wq, onnx.TensorProto.INT4)
        scale_tensor = make_gs_scale(name, scale)
        zp_tensor = make_gs_zp(name, scale.shape, onnx.TensorProto.INT4)
        dq_out = make_gs_dequantize_output(name, shape=wq.shape, dtype=scale.dtype)
        dq_node = make_gs_dequantize_node(
            name, inputs=[wq_tensor, scale_tensor, zp_tensor], outputs=[dq_out]
        )
        dq_nodes[name] = dq_node

    dq_nodes = {}
    for name, scale in scales.items():
        _insert_helper(name, quantized_weights[name], scale, dq_nodes)

    _postprocess_qdq(
        graph,
        orig_weight_names=set(scales.keys()),
        dq_nodes=dq_nodes,
    )


def insert_qdq_nodes(
    graph: gs.Graph,
    scales: Dict[str, np.ndarray],
    weight_map: Dict[str, gs.Tensor],
):
    """Insert scales and QDQ nodes into graph.

    Args:
        graph: The graph to modify.
        scales: A map from ONNX initializer name to desired scale factor for that initializer.
        weight_map: A map from ONNX initializer name to graphsurgeon tensor.
    """

    def _insert_helper(
        name: str,
        weight_to_quantize: gs.Tensor,
        scale: np.ndarray,
        q_nodes: Dict[str, gs.Node],
        dq_nodes: Dict[str, gs.Node],
    ):
        scale_tensor = make_gs_scale(name, scale)
        zp_tensor = make_gs_zp(name, scale.shape, onnx.TensorProto.INT4)
        q_out = make_gs_quantize_output(name, weight_to_quantize.shape, onnx.TensorProto.INT4)
        q_node = make_gs_quantize_node(
            name, inputs=[weight_to_quantize, scale_tensor, zp_tensor], outputs=[q_out]
        )
        dq_out = make_gs_dequantize_output(name, shape=weight_to_quantize.shape, dtype=scale.dtype)
        dq_node = make_gs_dequantize_node(
            name, inputs=[q_out, scale_tensor, zp_tensor], outputs=[dq_out]
        )
        q_nodes[name] = q_node
        dq_nodes[name] = dq_node

    q_nodes, dq_nodes = {}, {}
    for name, scale in scales.items():
        _insert_helper(name, weight_map[name], scale, q_nodes, dq_nodes)

    _postprocess_qdq(
        graph,
        orig_weight_names=set(scales.keys()),
        q_nodes=q_nodes,
        dq_nodes=dq_nodes,
    )


def replace_scale_values(graph: onnx.onnx_ml_pb2.GraphProto, act_scales_dict: Dict[str, float]):
    """Replaces the scales values from calibration cache."""
    initializers = graph.initializer

    def _get_initializer_index(name: str):
        for idx, init in enumerate(initializers):
            if init.name == name:
                return idx
        return None

    for node in graph.node:
        if node.op_type == "QuantizeLinear":
            scale_input_name = node.input[1]
            if scale_input_name in act_scales_dict:
                idx = _get_initializer_index(scale_input_name)
                scale = numpy_helper.from_array(
                    np.float32(act_scales_dict[scale_input_name]), scale_input_name
                )
                initializers[idx].CopyFrom(scale)
            else:
                # If the scale is not present in the act_scales_dict
                # then the current node must be an weight quantizer and
                # the weight should be available in the graph initializer
                assert (
                    _get_initializer_index(node.input[0]) is not None
                ), f"Tensor {node.input[0]} not found in initializers."
