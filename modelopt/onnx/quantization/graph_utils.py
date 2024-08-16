# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Provides ONNX graph related utils for QDQ placement."""
import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.ir.tensor import Constant, Tensor, Variable

from modelopt.onnx.op_types import is_copy_op, is_linear_op
from modelopt.onnx.utils import (
    find_lowest_common_ancestor,
    get_child_nodes,
    get_parent_nodes,
)


def is_const_input(tensor: Tensor) -> bool:
    """Returns whether the given tensor is an initializer or produced by const-foldable nodes."""
    if isinstance(tensor, Constant):
        return True

    # Tensor is a graph input variable
    if len(tensor.inputs) == 0:
        return False

    producer_node = tensor.inputs[0]  # Generally tensors has single producer
    if producer_node.op in ["Constant", "Identity"]:
        return True

    # Second axes input to Squeeze/Unsqueeze is a constant, we need to check the first input
    if producer_node.op in ["Squeeze", "Unsqueeze"] and is_const_input(producer_node.inputs[0]):
        return True

    # Const -> Clip -> Exp -> Mul pattern matching for swin_v2
    if producer_node.op == "Exp":
        clip_node = producer_node.i()
        if clip_node.op == "Clip" and has_const_input(clip_node):
            return True

    return False


def has_const_input(node: Node) -> bool:
    """Returns whether the given node has any constant input."""
    for tensor in node.inputs:
        if is_const_input(tensor):
            return True

    return False


def has_path_type(
    node: Node,
    graph: Graph,
    path_type: List[str],
    is_forward: bool,
    wild_card_types: List[str] = [],
    path_nodes: List[Node] = [],
) -> bool:
    """Checks if the given node is start/end of a given forward/backward path type.

    Note, Path can be forward or backward wrt a node depending on the next level nodes.
    Additionally, this method can work with optional nodes and collect the traversed path.

    Args:
        node: Start node of the path.
        graph: ONNX model graph.
        path_type: Path types to match from the given node.
        is_forward: Whether to match forward or backward path.
        wild_card_types: Wild card types, these type of nodes are skipped and not matched with the path_type.
        path_nodes: Accumulated nodes in the matched path.

    Returns:
        Bool, whether the given node is start/end of the given forward/backward path type.
    """
    optional_path_types = ["BiasAdd", "ConstMul"]
    if not path_type:
        # All types matched
        return True

    # Current node type and special type conversion for optional BiasAdd and ConstMul
    # Note, matching path with Add/Mul type nodes with const input will fail
    node_type = node.op
    if node_type == "Add" and has_const_input(node):
        node_type = "BiasAdd"
    elif node_type == "Mul" and has_const_input(node):
        node_type = "ConstMul"

    # Check if current non-wild node type does not match the expected path type
    # And if path type is not optional (ex. BiasAdd)
    is_match = (node_type == path_type[0]) or (node.op == path_type[0])
    is_wild_match = node_type in wild_card_types
    if not is_match and not is_wild_match and (path_type[0] not in optional_path_types):
        return False

    # Add current node name in the path
    if is_match:
        path_nodes.append(node)

    # If current node type matches the expected path type or path type is optional (ex. BiasAdd), we have a type match
    # Update the remaining path types to match
    next_path_type = path_type[:]

    # Non-repeatable optional types should be consumed
    if is_match or (path_type[0] in ["BiasAdd", "ConstMul"]):
        next_path_type = path_type[1:]

    # If current node is not wild card and didn't match, go ahead and match with the
    # remaining path types starting with the current node
    if not is_match and not is_wild_match:
        assert path_type[0] in optional_path_types
        return has_path_type(
            node,
            graph,
            next_path_type,
            is_forward,
            wild_card_types,
            path_nodes,
        )

    if is_forward:
        next_level_nodes = get_child_nodes(node)
    else:
        next_level_nodes = get_parent_nodes(node)

    # Check if any child (forward path) or parent (backward path) can match the remaining path types
    for next_node in next_level_nodes:
        sub_path = []
        if has_path_type(next_node, graph, next_path_type, is_forward, wild_card_types, sub_path):
            path_nodes.extend(sub_path)
            return True

    # Path type matches if there is no remaining types to match
    return not next_path_type


def get_fusible_backbone(node: Node, graph: Graph) -> Optional[Node]:
    """Returns the linear backbone node for a given node if it matches the pattern.

    TensorRT fuses convolution with BN, Relu etc. when in some specific pattern.
    This rule tries to match some of those patterns.
    Note. BiasAdd and ConstMul are optional in path types.

    Args:
        node: Start node of the pattern.
        graph: ONNX model graph.

    Returns:
        Backbone node of the given node, None if not found.
    """

    def _get_backbone(root: Node):
        if root.op == "Conv":
            return root

        for tensor in root.inputs:
            if not isinstance(tensor, Constant):
                parent_node = tensor.inputs[0]
                bb = _get_backbone(parent_node)
                if bb:
                    return bb

    fusible_linear_path_types = [
        # ["Sigmoid", "Conv"],  # With following Mul
        # ["Resize", "Relu", "Conv"],   # Note. this causes regression in MTL_v1
        ["BiasAdd", "ConstMul", "Conv"],
        ["Relu", "BiasAdd", "ConstMul", "Conv"],
        ["BatchNormalization", "BiasAdd", "Conv"],
        ["Relu", "BatchNormalization", "BiasAdd", "Conv"],
    ]
    for idx, path_type in enumerate(fusible_linear_path_types):
        if has_path_type(node, graph, path_type, is_forward=False, wild_card_types=[]):
            return _get_backbone(node)

    return None


def filter_quantizable_kgen_heads(
    cask_fusible_partitions: List[List[Node]],
    kgen_partitions: List[List[Node]],
    quantizable_op_types: List[str],
) -> Tuple[List[Node], List[Tuple[Node, Node, str]]]:
    """Returns the list of kgen head names if it follows a CASK partition."""
    cask_partition_nodes = set()
    for partition in cask_fusible_partitions:
        cask_partition_nodes.update([node.name for node in partition])

    cask_partition_heads = [partition[0] for partition in cask_fusible_partitions]

    def _is_following_cask_partition(node: Node):
        # Checking if cask fusible partition can be reached backward
        # ignoring the copy ops
        if node.name in cask_partition_nodes:
            return True

        if not is_copy_op(node.op):
            return False

        for parent in get_parent_nodes(node):
            if _is_following_cask_partition(parent):
                return True

        return False

    def _has_other_quantizable_consumer(
        tensor: Tensor, quantizable_kgen_heads: List[Node], head_name: str
    ):
        # Note. this is kinda approximate analysis,
        # all quantizable kgen heads may haven't got discovered yet
        quantizable_ops = [node.name for node in cask_partition_heads + quantizable_kgen_heads]

        # Look for other quantizable consumer than the current kgen head
        if head_name in quantizable_ops:
            quantizable_ops.remove(head_name)

        for consumer in tensor.outputs:
            if consumer.name in quantizable_ops:
                return True

        return False

    quantizable_kgen_heads = []
    no_quantize_inputs = []  # list of tuple [(src_node_name, dst_node_name, input_name), ...]
    output_quantization_candidates = [
        "AveragePool",
        "BatchNormalization",
        "GlobalAveragePool",
        "MaxPool",
    ]

    for partition in kgen_partitions:
        head_node = partition[0]
        # Check if partition head is of default quantizable type
        if head_node.op not in quantizable_op_types:
            continue

        # If the node has cost input, do not quantize
        if has_const_input(head_node):
            continue

        head_parents = get_parent_nodes(head_node)
        no_quantize_inputs_of_head = []
        has_quantizable_input = False

        # Check each of the parent (input producer for partition head)
        # or predecessor nodes and see if output quantization is needed for them
        # and decide which input of kgen head needs quantization
        for parent in head_parents:
            # If the head is consuming output of any quantizable op, then it is quantizable
            if _is_following_cask_partition(parent) or parent.op in output_quantization_candidates:
                quantizable_kgen_heads.append(partition[0])
                has_quantizable_input = True
            # If the input from the current parent has no other quantizable consumer, do not quantize that input
            elif not _has_other_quantizable_consumer(
                parent.outputs[0], quantizable_kgen_heads, head_node.name
            ):
                no_quantize_inputs_of_head.append((parent, partition[0], parent.outputs[0].name))

        # If at least one input of Add is quantizable, collect if there is any non-quantizable inputs
        if head_node.op == "Add" and has_quantizable_input:
            no_quantize_inputs.extend(no_quantize_inputs_of_head)

    return quantizable_kgen_heads, no_quantize_inputs


def classify_partition_nodes(
    partitions: List[List[Node]],
) -> Tuple[List[Node], List[Node], List[Tuple[Node, Node, str]]]:
    """We should partially quantize the partition nodes with inputs outside of the partition.

    Args:
        partitions: Partitions created by modelopt ptq algo.

    Returns:
        List of non-quantizable nodes.
        List of quantizable nodes.
        List of partially-quantizable inputs with non-quantizable input info as (src, dst, input_name)
    """
    non_quantizable_partition_nodes = []  # list of Node [node1, ...]
    quantizable_partition_nodes = []  # list of Node [node1, ...]
    no_quantize_inputs = []  # list of tuple [(src_node, dst_node, input_name), ...]

    for partition in partitions:
        partition_root_type = partition[0].op
        assert is_linear_op(partition_root_type)

        # Collect tensor names produced by partition nodes
        partition_node_outputs = []
        for node in partition:
            for node_output in node.outputs:
                partition_node_outputs.append(node_output.name)

        for node in partition:
            has_external_inputs = False
            internal_inputs = []  # Keeps (producer, consumer, tensor)
            for tensor in node.inputs:
                if is_const_input(tensor):
                    continue

                # If a KGEN op has external non-constant input, it is considered partially quantizable
                if tensor.name not in partition_node_outputs:
                    # partition heads will be fully quantizable and added
                    has_external_inputs = True
                else:
                    producer_node = tensor.inputs[0]
                    # format: source, target, input
                    # Note. it might happen that this node was not quantized
                    # We just ignore it from no_quantize_inputs list in post-processing
                    internal_inputs.append((producer_node, node, tensor.name))

            if not has_external_inputs:
                non_quantizable_partition_nodes.append(node)
            elif has_external_inputs and internal_inputs:
                no_quantize_inputs.extend(internal_inputs)
            else:
                # partition head is quantizable
                quantizable_partition_nodes.append(node)

    return non_quantizable_partition_nodes, quantizable_partition_nodes, no_quantize_inputs


def build_non_residual_input_map(graph: Graph) -> Dict[str, str]:
    """Builds a map of non-residual Add input name to the Add node name from the given graph.

    This assumes that the Add layer only has 2 inputs.

    We will refer to a subgraph which has a Convolution node with a single output that is summed (element-wise)
    with another non-constant input-tensor as a “residual-add” subgraph, because it occurs in modern
    convnets that use residual connections.

    Args:
        graph: Onnx model graph.

    Returns:
        Dictionary of Add node names vs their non-residual input name.
    """
    non_residual_inputs = {}
    for node in graph.nodes:
        if node.op in ["Add"]:
            # Add nodes with constant or graph input does not have non-residual input
            # Here, A = node.inputs[0], B = node.inputs[1] and A.inputs means producer nodes of A
            # TODO: make this check a util?
            if (
                has_const_input(node)
                or len(node.inputs[0].inputs) == 0
                or len(node.inputs[1].inputs) == 0
            ):
                non_residual_inputs[node.name] = None
                continue

            input1_producer = node.i(0, 0)
            input2_producer = node.i(1, 0)

            backbone1 = get_fusible_backbone(input1_producer, graph)
            backbone2 = get_fusible_backbone(input2_producer, graph)

            # Generally if both the inputs have a backbone then both backbones are of the same type
            if backbone1 and backbone2:
                if backbone1 == backbone2:
                    non_residual_inputs[node.name] = None
                    continue

                assert backbone1.op == backbone2.op, (
                    f"{backbone1.name} and {backbone2.name} are different types of backbone for"
                    f" {node.name}!"
                )
                # Input in the longest path to LCA is the non-residual input
                _, d1, d2 = find_lowest_common_ancestor(input1_producer, input2_producer)
                if d1 > d2:
                    non_residual_inputs[node.name] = node.inputs[0].name
                else:
                    non_residual_inputs[node.name] = node.inputs[1].name
            elif backbone1:
                non_residual_inputs[node.name] = node.inputs[0].name
            elif backbone2:
                non_residual_inputs[node.name] = node.inputs[1].name
            else:
                # Not a residual Add node
                non_residual_inputs[node.name] = None

    return non_residual_inputs


def remove_partial_input_qdq(
    graph: Graph,
    no_quantize_inputs: List[Tuple[Node, Node, str]],
) -> None:
    """Modifies the onnx model by removing QDQ nodes from the marked inputs, ex. non-residual inputs etc.

    Args:
        graph: Onnx model graph.
        no_quantize_inputs: List non-quantizable input info as (src, dst, input_name)
    """
    logging.info("Deleting QDQ nodes from marked inputs to make certain operations fusible ...")
    graph_nodes = {node.name: node for node in graph.nodes}
    for source, target, non_qdq_input_name in no_quantize_inputs:
        # Note. no_quantize_inputs objects are from non-quantized input graph
        # we are deleting some QDQ from the new quantized output graph
        source_node = graph_nodes[source.name]
        try:
            dq_node = source_node.o().o()
        except Exception:
            # Reached end of the graph
            continue
        if dq_node.op == "DequantizeLinear":
            dq_node = dq_node.outputs[0]  # source_node->Q->DQ->target_node
            while len(dq_node.outputs):
                # Find the input index in the target connecting with source_node
                target_input_idx_arr = [
                    idx
                    for idx, inp in enumerate(dq_node.outputs[0].inputs)
                    if inp.name == dq_node.name
                ]
                target_input_idx = target_input_idx_arr[0] if target_input_idx_arr else 0

                # Connect the output of source_node with the outputs of DQ until DQ is not connected to any other
                #   layers. Note that when a connection is removed, this is also deleted from dq_node.outputs, thus
                #   why we keep iterating over the same idx=0 in dq_node.outputs[0].
                dq_node.outputs[0].inputs[target_input_idx] = source_node.outputs[0]

    graph.cleanup()
    graph.toposort()


def _find_nodes_from_op_types_to_exclude(graph: Graph, op_types_to_exclude=None) -> List[str]:
    nodes_to_exclude = []
    if op_types_to_exclude:
        nodes_to_exclude = [node.name for node in graph.nodes if node.op in op_types_to_exclude]
    return nodes_to_exclude


def _expand_node_names_from_patterns(graph: Graph, name_patterns: List[str]) -> List[str]:
    matched_node_names = []
    for pattern in name_patterns:
        for node in graph.nodes:
            if re.match(pattern, node.name):
                matched_node_names.append(node.name)

    return matched_node_names


def find_nodes_to_exclude(
    graph: Graph, nodes_to_exclude: List[str], op_types_to_exclude: List[str]
):
    """Find the node names from the ONNX graph which matches user's exclusion patterns."""
    nodes_to_exclude = nodes_to_exclude or []
    nodes_to_exclude = _expand_node_names_from_patterns(graph, nodes_to_exclude)
    nodes_to_exclude.extend(_find_nodes_from_op_types_to_exclude(graph, op_types_to_exclude))

    # Remove duplicates from the exclusion list
    return [*set(nodes_to_exclude)]


def add_fp16_fp32_cast(onnx_path, custom_ops_to_cast_to_fp16):
    """Adds cast_to_fp16 nodes to the inputs of a layer and cast_to_fp32 to the outputs."""
    name_dict = {}

    def _get_unique_name(old_name):
        if old_name not in name_dict:
            name_dict[old_name] = 0
            return old_name
        name_dict[old_name] = name_dict[old_name] + 1
        return old_name + "_" + str(name_dict[old_name])

    def _add_cast_node(tensor, precision="fp16"):
        if precision == "fp16":
            onnx_precision = int(onnx.TensorProto.FLOAT16)
            np_precision = "float16"
        else:
            onnx_precision = int(onnx.TensorProto.FLOAT)
            np_precision = "float32"

        cast_out = Variable(
            name=_get_unique_name(tensor.name + f"_cast_to_{precision}_in"),
            dtype=np_precision,
            shape=tensor.shape,
        )
        cast_node = Node(
            op="Cast",
            name=_get_unique_name(tensor.name + f"_cast_to_{precision}"),
            attrs={"to": onnx_precision},
            inputs=[tensor],
            outputs=[cast_out],
        )
        graph.nodes.append(cast_node)
        return cast_out

    graph = gs.import_onnx(onnx.load(onnx_path))
    castable_nodes = [n for n in graph.nodes if n.op in custom_ops_to_cast_to_fp16]

    for node in castable_nodes:
        # Cast all inputs to FP16
        for inp_idx, inp in enumerate(node.inputs):
            cast_out = _add_cast_node(inp, precision="fp16")
            node.inputs[inp_idx] = cast_out

        # Cast all outputs back to FP32
        for out in node.outputs:
            cast_out = _add_cast_node(out, precision="fp32")
            node.o().inputs[0] = cast_out

    graph.cleanup().toposort()

    new_onnx_path = onnx_path.replace(".onnx", "_castFP16.onnx")
    onnx.save(gs.export_onnx(graph), new_onnx_path)
    return new_onnx_path


def print_stat(graph: Graph, verbose: bool) -> None:
    """Collect and print stats of the quantized model."""
    count = 0
    quantized_node_types = set()
    quantized_nodes = []
    output_names = [output_node.name for output_node in graph.outputs]
    for node in graph.nodes:
        for tensor in node.inputs:
            if len(tensor.inputs) == 0:
                continue

            producer_node = tensor.inputs[0]
            if producer_node.op == "DequantizeLinear":
                quantized_node_types.add(node.op)
                quantized_nodes.append(node.name)
                count += 1
                break
            else:
                # Sometimes "_DequantizeLinear_Output" is not suffix of the "DequantizeLinear" typed node,
                # if that node is also in final model output. Ex. CLIP-ViT-L-14-opset16.onnx
                assert tensor.name in output_names or producer_node.op != "DequantizeLinear"

    if verbose:
        logging.info(f"Quantized nodes: {quantized_nodes}")
    logging.info(f"Total number of quantized nodes: {count}")
    logging.info(f"Quantized node types: {quantized_node_types}")


def find_mha_partitions(graph):
    """Match MHA: BMM1 -> (Mul/Div) -> (Add) -> Softmax -> (Cast) -> BMM2."""
    mha_chain_type = ["MatMul", "Softmax", "MatMul"]
    wild_card_types = [
        "Div",
        "Mul",
        "ConstMul",
        "Add",
        "BiasAdd",
        "Reshape",
        "Transpose",
        "Flatten",
        "Cast",
    ]
    mha_partitions = []
    for node in graph.nodes:
        if node.op == "MatMul":
            mha_partition = []
            if has_path_type(node, graph, mha_chain_type, True, wild_card_types, mha_partition):
                if (
                    len(mha_partition) == 3
                    and mha_partition[0].op == "MatMul"
                    and mha_partition[2].op == "MatMul"
                ):
                    mha_partitions.append(mha_partition)

    return mha_partitions


def find_fp8_mha_partitions(graph):
    """Match FP8 MHA: Q -> DQ -> BMM1 -> (Mul/Div) -> (Add) -> Softmax -> (Cast) -> Q -> DQ -> BMM2 -> Q -> DQ."""
    softmax_bmm1_chain_type = ["Softmax", "MatMul", "DequantizeLinear", "QuantizeLinear"]
    softmax_bmm2_chain_type = [
        "Softmax",
        "QuantizeLinear",
        "DequantizeLinear",
        "MatMul",
        "QuantizeLinear",
        "DequantizeLinear",
    ]
    wild_card_types = [
        "Div",
        "Mul",
        "ConstMul",
        "Add",
        "BiasAdd",
        "Reshape",
        "Transpose",
        "Flatten",
        "Cast",
    ]
    fp8_mha_partitions = []
    for node in graph.nodes:
        if node.op == "Softmax":
            fp8_mha_partition = []
            if has_path_type(
                node, graph, softmax_bmm1_chain_type, False, wild_card_types, fp8_mha_partition
            ) and has_path_type(
                node, graph, softmax_bmm2_chain_type, True, wild_card_types, fp8_mha_partition
            ):
                if (
                    len(fp8_mha_partition) == 10
                    and fp8_mha_partition[1].op == "MatMul"
                    and fp8_mha_partition[7].op == "MatMul"
                ):
                    fp8_mha_partitions.append(fp8_mha_partition)

    return fp8_mha_partitions


def insert_matmul_casts(graph, matmul_node):
    """Insert three cast nodes for MatMul's two inputs and output."""
    matmul_input0 = matmul_node.inputs[0]
    matmul_input0_cast_output = gs.Variable(
        name=f"{matmul_input0.name}/Cast_output", dtype=np.float32
    )
    graph.layer(
        op="Cast",
        name=f"{matmul_input0.name}/Cast",
        inputs=[matmul_input0],
        outputs=[matmul_input0_cast_output],
        attrs={"to": np.float32},
    )
    matmul_node.inputs[0] = matmul_input0_cast_output

    matmul_input1 = matmul_node.inputs[1]
    matmul_input1_cast_output = gs.Variable(
        name=f"{matmul_input1.name}/Cast_output", dtype=np.float32
    )
    graph.layer(
        op="Cast",
        name=f"{matmul_input1.name}/Cast",
        inputs=[matmul_input1],
        outputs=[matmul_input1_cast_output],
        attrs={"to": np.float32},
    )
    matmul_node.inputs[1] = matmul_input1_cast_output

    matmul_output = matmul_node.outputs[0]
    matmul_output_cast_input = gs.Variable(
        name=f"{matmul_output.name}/Cast_output", dtype=np.float16
    )
    graph.layer(
        op="Cast",
        name=f"{matmul_output.name}/Cast",
        inputs=[matmul_output_cast_input],
        outputs=[matmul_output],
        attrs={"to": np.float16},
    )
    matmul_node.outputs[0] = matmul_output_cast_input


def insert_fp8_mha_casts(onnx_model):
    r"""Insert three cast ops.

    The first cast will be added before the input0 of MatMul to cast fp16 to fp32.
    The second cast will be added before the input1 of MatMul to cast fp16 to fp32.
    The third cast will be added after the output of MatMul to cast fp32 back to fp16.
    The insertion of Cast ops in the FP8 MHA part actually forbids the MHAs to run
    with FP16 accumulation because Myelin only has FP32 accumulation kernels for FP8 MHAs.
    """
    graph = gs.import_onnx(onnx_model)
    graph.cleanup().toposort()

    # Match FP8 MHA: Q -> DQ -> BMM1 -> (Mul/Div) -> (Add) -> Softmax -> (Cast) -> Q -> DQ -> BMM2 -> Q -> DQ
    fp8_mha_partitions = find_fp8_mha_partitions(graph)

    # Insert cast nodes on BMM1 and BMM2's input and output tensors.
    for fp8_mha_partition in fp8_mha_partitions:
        insert_matmul_casts(graph, fp8_mha_partition[1])
        insert_matmul_casts(graph, fp8_mha_partition[7])

    graph.cleanup().toposort()

    return gs.export_onnx(graph)
