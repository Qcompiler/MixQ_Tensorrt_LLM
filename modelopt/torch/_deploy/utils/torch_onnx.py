# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utility functions related to Onnx."""
import inspect
import os
import shutil
from typing import Any, Dict, List, Tuple, Union

import cloudpickle
import onnx
import torch
import torch.nn as nn
from onnx import ModelProto
from torch.nn.parallel import DataParallel, DistributedDataParallel

from modelopt.onnx.utils import (
    get_input_names,
    get_input_shapes,
    get_node_names,
    get_output_names,
    get_output_shapes,
)
from modelopt.torch.utils import flatten_tree, standardize_named_model_args
from modelopt.torch.utils._pytree import TreeSpec

from .._runtime.common import write_bytes
from .onnx_optimizer import Optimizer

ModelMetadata = Dict[str, Any]
ModelType = Any
ValueInfoType = Any

# a few constants...
DEFAULT_ONNX_OPSET = 13
ONNX_EXPORT_IN_PREFIX = "in"
ONNX_EXPORT_OUT_PREFIX = "out"
TWO_GB = 2147483648


class OnnxBytes:
    """A class to save and load onnx models as bytes."""

    def __init__(self, onnx_load_path: str, external_data_format: bool = False) -> None:
        """Loads the model from the specified path.

        If the model is loaded without external data format, then it is saved as a dictionary where
        the key is the model name and the value is the model bytes.
        If the model is loaded with external data format, then the model is saved as a dictionary
        where the keys include all the file names in the model directory and the value are the corresponding file bytes.
        For external data format, we assume that the external data for the model is saved in the same directory
        as the model file.

        Args:
            onnx_load_path: The path to load the .onnx model file.
            external_data_format: If True, the onnx model is loaded from the external data format.
        """
        self.onnx_model = {}
        self.model_name = ""
        print("Loading onnx model from path:", onnx_load_path)
        if external_data_format:
            onnx_load_path = os.path.dirname(onnx_load_path)
            for onnx_model_file in os.listdir(onnx_load_path):
                with open(os.path.join(onnx_load_path, onnx_model_file), "rb") as f:
                    self.onnx_model[onnx_model_file] = f.read()
                if onnx_model_file.endswith(".onnx"):
                    if self.model_name != "":
                        raise ValueError("Multiple onnx files found in the directory")
                    self.model_name = onnx_model_file.replace(".onnx", "")
        else:
            onnx_model_file = os.path.basename(onnx_load_path)
            if not onnx_model_file.endswith(".onnx"):
                raise ValueError("The file should be a .onnx file")
            with open(onnx_load_path, "rb") as f:
                self.onnx_model[onnx_model_file] = f.read()
            self.model_name = onnx_model_file.replace(".onnx", "")

    def write_to_disk(self, onnx_save_path: str) -> None:
        """Writes the onnx model to the specified path."""
        if os.path.exists(onnx_save_path):
            print(f"Removing existing directory: {onnx_save_path}")
            shutil.rmtree(onnx_save_path)
        os.makedirs(onnx_save_path)
        print("Writing onnx model to path:", onnx_save_path)
        for onnx_model_file, onnx_model_bytes in self.onnx_model.items():
            with open(os.path.join(onnx_save_path, onnx_model_file), "wb") as f:
                f.write(onnx_model_bytes)

    def to_bytes(self) -> bytes:
        """Returns the bytes of the object."""
        return cloudpickle.dumps(self)

    def get_onnx_model_file_bytes(self) -> bytes:
        """Returns the bytes of the onnx model file."""
        return self.onnx_model[self.model_name + ".onnx"]

    @classmethod
    def from_bytes(cls, onnx_bytes: bytes) -> "OnnxBytes":
        """Returns the OnnxBytes object from the bytes."""
        return cloudpickle.loads(onnx_bytes)


def _to_expected_onnx_type(val: Any) -> Any:
    """Convert the given value to the expected onnx type.

    During the onnx export process, plain numeric types (floats and ints) are converted to torch
    tensors. This function pre-converts the given val to a tensor in case val is a int or float for
    easier handling of such input values during the onnx export process.
    """
    if isinstance(val, (int, float)):
        return torch.tensor(val).to(type(val))
    return val


def generate_onnx_input(model_metadata: ModelMetadata, input: Union[Any, Tuple]) -> Dict[str, Any]:
    """Generate input for onnx model from model's forward signature and provided input.

    Args:
        model_metadata: The model's metadata.
        input: A tuple of args/kwargs or torch.Tensor feed into the model's ``forward()`` method,
            see :meth:`standardize_model_args() <modelopt.torch.utils.network.standardize_model_args>`
            for more info on the convention.

    Returns:
        Args flattened into one dictionary with serialized keys compatible with provided onnx.

    .. note::

        This function performs a sanity check on the provided input data to filter out args that
        are constants (instead of input nodes) in the onnx graph.


    Some more relevant background of why we want to flatten the input pytree here:

        * In the onnx export process, nested python data structures (like nested lists, tuples,
            dictionaries) are being recursed into until leaf objects corresponding to tensors are
            encountered.

        * This is used to flatten the input in an onnx model to a list of tensors.

        * However, this is a fairly complex process for the user to understand in case their models
            takes a nested data structure. They have to understand how to manually flatten the data
            structure in the *correct* order in order for them to run inference on a device_model or
            onnx model.

        * With this function this additional complexity can be abstracted away from the user.

        * Example: if the original model took ``[x, {"y":y, "z" : [z1,z2]}]`` they can still provide
            this nested data structure instead of the expected onnx input list of ``[x, y, z1, z2]``
            --> flattening and unflattering is handled internally.
    """
    # get named args and set of params where we added default values
    named_args, args_with_default = standardize_named_model_args(model_metadata["signature"], input)

    # retrieve onnx input names
    onnx_input_names = model_metadata["input_onnx_names"]
    input_none_names = model_metadata["input_none_names"]

    # capture flattened names of args from default values
    named_default_args = {k: v for k, v in named_args.items() if k in args_with_default}
    _, tree_spec_default_args = flatten_tree(named_default_args, prefix=ONNX_EXPORT_IN_PREFIX)

    # capture flattened args without default args that do not appear in onnx graph
    values, tree_spec = flatten_tree(named_args, prefix=ONNX_EXPORT_IN_PREFIX)
    flat_kv = {k: v for k, v in zip(tree_spec.names, values)}

    # We wanna consider four types of flattened args:
    # 1. Args that appear in the onnx graph
    # 2. Args that are not their default value
    # 3. Args that were provided as None during conversion but are not None right now
    # 4. Args that were provided as None during conversion and are None right now

    args_in_onnx = {k for k in flat_kv if k in onnx_input_names}
    args_not_default = {k for k in flat_kv if k not in tree_spec_default_args.names}
    args_not_none = {k for k, v in flat_kv.items() if k in input_none_names and v is not None}
    args_none = {k for k, v in flat_kv.items() if k in input_none_names and v is None}

    # identify unexpected args from these 4 types
    unexpected_args = ((args_not_default - args_none) | args_not_none) - args_in_onnx
    if unexpected_args:
        raise ValueError(
            "The following args were provided that do not appear in the onnx graph of your model "
            "since they are treated as constants in the onnx graph:"
            + "\t\n".join(unexpected_args)
            + "\nConsider removing these args from your input that are constants in the onnx model "
            "or recompiling your onnx model with new constant values!"
        )

    # return the args that are relevant for the onnx graph in the right type
    return {k: _to_expected_onnx_type(v) for k, v in flat_kv.items() if k in args_in_onnx}


def optimize(name, onnx_graph, verbose=False):
    """Optimizes onnx graph."""
    opt = Optimizer(onnx_graph, verbose=verbose)
    opt.info(name + ": original")
    opt.cleanup()
    opt.info(name + ": cleanup")
    # TODO: fold constants is not working for some models from deploy_models(NestedOutModel, ArgsKwargsModel1)
    # opt.fold_constants()
    # opt.info(name + ": fold_constants")
    onnx_graph = opt.infer_shapes(return_onnx=True)
    opt.info(name + ": shape inference")
    return onnx_graph


def get_onnx_bytes_and_metadata(
    model: nn.Module,
    dummy_input: Union[Any, Tuple],
    onnx_load_path: str = "",
    remove_exported_model: bool = True,
    **kwargs,
) -> Tuple[bytes, ModelMetadata]:
    """Get onnx model in bytes from input pytorch model together with the input/output of model.

    Arguments:
        model: PyTorch model to export to onnx.
        dummy_input: A tuple of args/kwargs or torch.Tensor, see
            `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_
            for more info on the convention.
        onnx_load_path: The path to load the onnx model.
        remove_exported_model: If True, the onnx model will be cleared from the disk after the
            export process
        **kwargs: Additional arguments to pass, e.g., ``onnx_opset``.

    Returns:
        bytes: Onnx model in bytes.
        ModelMetadata: The model's meta data.

    Raises:
        ValueError: If nn.Module is not passed as model.
    """
    if not isinstance(model, nn.Module):
        raise ValueError("Only PyTorch model compilation is supported.")

    # unwrap DDP and DP models
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module

    # process onnx_opset
    onnx_opset = int(kwargs.get("onnx_opset", DEFAULT_ONNX_OPSET))

    # Standardize model args and also tensorize them so they also appear in the onnx graph!
    # Floats/ints are tensorized when they are provided, but not tensorized when they are not
    # provided which is somewhat inconsistent (we always tensorize them!)
    named_args, _ = standardize_named_model_args(model, dummy_input)
    named_args = {k: _to_expected_onnx_type(v) for k, v in named_args.items()}

    # Also standardize dummy_input again so we can use it
    dummy_input = tuple(named_args.values())
    if dummy_input and isinstance(dummy_input[-1], dict):
        dummy_input = dummy_input + ({},)  # we need to add an extra dict for the fake kwargs!

    # Get input tree spec, see generate_onnx_input for more info as well on this
    flat_input, tree_spec_input = flatten_tree(named_args, prefix=ONNX_EXPORT_IN_PREFIX)

    # input names are the names of the flattened input tree spec but without None values
    input_names = [k for k, v in zip(tree_spec_input.names, flat_input) if v is not None]

    # we also want to record the input names that are None so we can remove them from the input
    # during inference.
    input_none_names = list(set(tree_spec_input.names) - set(input_names))

    # Get output once (we export in eval mode - so also using eval mode here!)
    is_training = model.training
    model.eval()
    output = model(*named_args.values())
    model.train(is_training)

    # Get output tree spec
    flat_output, tree_spec_output = flatten_tree(output, prefix=ONNX_EXPORT_OUT_PREFIX)

    # output names are the names of the flattened input tree spec but without None values
    output_names = [k for k, v in zip(tree_spec_output.names, flat_output) if v is not None]

    model_name = model.__class__.__name__
    onnx_build_folder = "./build/onnx/"
    onnx_path = os.path.join(onnx_build_folder, model_name)
    os.makedirs(onnx_path, exist_ok=True)
    onnx_save_path = os.path.join(onnx_path, f"{model_name}.onnx")

    # If the onnx_load path is specified by the user or if an onnx model exists at the default path
    # then the model is loaded from this path and returned along with the metadata
    if os.path.exists(onnx_save_path):
        print(f"Overriding onnx load path to {onnx_save_path}")
        onnx_load_path = onnx_save_path

    if onnx_load_path != "":
        onnx_model = OnnxBytes(onnx_load_path)
        onnx_model_graph = onnx.load(os.path.join(onnx_load_path))
        model_metadata = create_model_metadata(
            tree_spec_input, tree_spec_output, input_none_names, onnx_model_graph, model
        )
        return onnx_model.to_bytes(), model_metadata

    # Export onnx model from pytorch model
    # As the maximum size of protobuf is 2GB, we cannot use io.BytesIO() buffer during export.
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_save_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=onnx_opset,
        )

    # Check that export worked
    assert len(os.listdir(onnx_path)) > 0, "Torch to onnx export failed."

    # Load the onnx graph for optimizaiton
    onnx_graph = onnx.load(onnx_save_path)

    # Optimize the onnx graph
    onnx_opt_graph = optimize(model.__class__.__name__, onnx_graph)

    # Remove training_mode attribute from BatchNormalization nodes
    onnx_opt_graph = remove_node_training_mode(onnx_opt_graph, "BatchNormalization")

    model_metadata = create_model_metadata(
        tree_spec_input, tree_spec_output, input_none_names, onnx_opt_graph, model
    )

    # Change the ONNX IR version to 9 to be compatible with ONNXRuntime
    onnx_opt_graph.ir_version = 9

    # If the onnx model is larger than 2GB, we need to save it in multiple files using the
    # external data format.
    if onnx_opt_graph.ByteSize() > TWO_GB:
        onnx.save_model(
            onnx_opt_graph,
            onnx_save_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            convert_attribute=False,
        )
    else:
        onnx.save(onnx_opt_graph, onnx_save_path)
    onnx_bytes = OnnxBytes(onnx_save_path)

    if remove_exported_model:
        shutil.rmtree(os.path.dirname(onnx_build_folder))
    return onnx_bytes.to_bytes(), model_metadata


def create_onnx_model_dict(
    onnx_load_path: str = None, model_metadata: ModelMetadata = None, model_dict: dict = None
) -> Tuple[bytes, ModelMetadata]:
    """Creates onnx model dictionary and model metadata.

    Args:
        onnx_load_path: The path to load the onnx model.
        model_metadata: Metadata to be stored in the DeviceModel.
        model_dict: A dictionaty containing the Pytorch model and the input and output treespec for the model.

    Returns:
        A tuple of onnx model dictionary and model metadata.
    """
    onnx_model_dict = {"onnx_load_path": onnx_load_path}
    onnx_model = {}
    for onnx_model_file in os.listdir(onnx_load_path):
        with open(os.path.join(onnx_load_path, onnx_model_file), "rb") as f:
            onnx_model[onnx_model_file] = f.read()
        if onnx_model_file.endswith(".onnx") and model_metadata is None:
            onnx_model_file_path = os.path.join(onnx_load_path, onnx_model_file)
            onnx_graph = onnx.load(onnx_model_file_path)
            onnx_input_names = [input.name for input in onnx_graph.graph.input]
            torch_input_names = model_dict["tree_spec_input"].names
            assert set(onnx_input_names).issubset(
                set(torch_input_names)
            ), "One or more inputs in the onnx model are not present in the torch model."
    onnx_model_dict["onnx_model"] = onnx_model
    if model_metadata is None:
        model_metadata = create_model_metadata(
            model_dict["tree_spec_input"],
            model_dict["tree_spec_output"],
            model_dict["input_none_names"],
            onnx_graph,
            model_dict["model"],
        )
    return cloudpickle.dumps(onnx_model_dict), model_metadata


def create_model_metadata(
    tree_spec_input: TreeSpec,
    tree_spec_output: TreeSpec,
    input_none_names: List[str],
    onnx_graph: ModelProto,
    model: nn.Module,
) -> ModelMetadata:
    """Create model metadata from the given input.

    Args:
        tree_spec_input: pytree spec describing the structure of the pytree for the model input.
        tree_spec_output: pytree spec describing the structure of the pytree for the model output.
        input_none_names: List of input names with values that are None.
        onnx_opt_graph: Graph of the onnx model.
        model: Pytorch model.

    Returns:
        ModelMetadata: The DeviceModel metadata.
    """
    return {
        "input_tree_spec": tree_spec_input,
        "input_shapes": get_input_shapes(onnx_graph),
        "input_onnx_names": get_input_names(onnx_graph),
        "input_none_names": input_none_names,
        "output_tree_spec": tree_spec_output,
        "output_shapes": get_output_shapes(onnx_graph),
        "output_onnx_names": get_output_names(onnx_graph),
        "signature": inspect.signature(model.forward),
        "onnx_node_names": get_node_names(onnx_graph),
        "is_bytes_pickled": onnx_graph.ByteSize() > TWO_GB,
    }


def write_onnx_bytes(onnx_model_bytes: Union[bytes, dict], onnx_save_path: str) -> Tuple[str, str]:
    """Write onnx bytes to the specified path.

    Args:
        onnx_model_bytes: Can be one of the following:
            1. A pickled dictionary of onnx model files. The keys of this dictionary are file names and\
            the values are the corresponding bytes.
            2. A single onnx model bytes.
        onnx_save_path: The onnx path to save the model.

    Returns:
        The path to the onnx model.
        Name of the onnx model.
    """
    if isinstance(onnx_model_bytes, dict):
        if not os.path.exists(onnx_save_path):
            os.makedirs(onnx_save_path)
        for key in onnx_model_bytes["onnx_model"]:
            write_bytes(onnx_model_bytes["onnx_model"][key], os.path.join(onnx_save_path, key))
        onnx_model_file = [s for s in onnx_model_bytes["onnx_model"].keys() if s.endswith(".onnx")]
        assert len(onnx_model_file) == 1, "There should be only one onnx model file"
        model_name = onnx_model_file[0].split(".")[0]
        return os.path.join(onnx_save_path, onnx_model_file[0]), model_name
    write_bytes(onnx_model_bytes, os.path.join(onnx_save_path, "model.onnx"))
    return os.path.join(onnx_save_path, "model.onnx"), ""


def get_onnx_bytes(*args, **kwargs):
    """Return onnx bytes only.

    See ``get_onnx_bytes_and_metadata()`` for more info.
    """
    onnx_bytes = get_onnx_bytes_and_metadata(*args, **kwargs)[0]
    onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
    return onnx_bytes_obj.onnx_model[f"{onnx_bytes_obj.model_name}.onnx"]


def remove_node_training_mode(onnx_model: ModelProto, node_op_type: str) -> ModelProto:
    """Remove training_mode attribute from selected node type.

    Args:
        onnx_model: The onnx model.
        node_op_type: The node type to remove training_mode attribute from.

    Returns:
        The onnx model with the training_mode attribute removed.
    """
    for node in onnx_model.graph.node:
        if node.op_type == node_op_type:
            for attribute in node.attribute:
                if attribute.name == "training_mode":
                    if attribute.i == 1:
                        node.output.remove(node.output[1])
                        node.output.remove(node.output[1])
                    attribute.i = 0

    return onnx_model
