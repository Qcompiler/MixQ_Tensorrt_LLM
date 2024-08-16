# Inspired by https://github.com/ELS-RD/transformer-deploy/blob/6b88e24ade6ce199e825adc0477b28a07f51f17d/src/transformer_deploy/QDQModels/ast_operator_patch.py

# Apache License
# Copyright 2022, Lefebvre Dalloz Services

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Support quantization for KV Cache in attention layers."""
import ast
import inspect
import sys
import tempfile
import types

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer

__all__ = ["register_attention_for_kv_quant"]


def register_attention_for_kv_quant(attention_cls: type) -> bool:
    """Register attention layer for quantization of KV Cache.

    Generate a quantized version of the attention class on the fly,
    and register it with the original class for quantization.
    """
    python_version = sys.version_info
    if not python_version >= (3, 9):
        print(f"Found {python_version.major}.{python_version.minor}.{python_version.micro}")
        raise RuntimeError("Python version >= 3.9 is required for KV Cache quantization")

    source_code = inspect.getsource(attention_cls)
    model_module = inspect.getmodule(attention_cls)
    head = ast.parse(source_code)

    bmm_ops = ("matmul", "bmm", "baddbmm")
    sdpa_ops = ("scaled_dot_product_attention",)

    def is_bmm(node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            # and isinstance(node.func.value, ast.Name)
            # and node.func.value.id == "torch"
            and node.func.attr in bmm_ops
        )

    def is_sdpa(node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in sdpa_ops
        )

    def is_bin_matmul(node):
        return isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult)

    def patch(node, quantizer_names):
        for index, quantizer_name in enumerate(quantizer_names):
            if quantizer_name is None:
                continue
            arg = node.args[index]
            node.args[index] = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()), attr=quantizer_name, ctx=ast.Load()
                ),
                args=[arg],
                keywords=[],
            )

    def patch_binop(node, quantizer_names):
        assert len(quantizer_names) == 2
        if quantizer_names[0] is not None:
            node.left = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=quantizer_names[0],
                    ctx=ast.Load(),
                ),
                args=[node.left],
                keywords=[],
            )
        if quantizer_names[1] is not None:
            node.right = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=quantizer_names[1],
                    ctx=ast.Load(),
                ),
                args=[node.right],
                keywords=[],
            )

    nodes = list(ast.walk(head))
    org_class_name = nodes[1].name  # type: ignore[attr-defined]
    new_class_name = nodes[1].name = "_Quant" + nodes[1].name  # type: ignore[attr-defined]

    bmm_nodes = []
    sdpa_nodes = []
    bin_matmul_nodes = []
    for node in ast.walk(head):
        if is_bmm(node):
            bmm_nodes.append(node)
        if is_sdpa(node):
            sdpa_nodes.append(node)
        if is_bin_matmul(node):
            bin_matmul_nodes.append(node)

    if len(bmm_nodes) != 2 and len(sdpa_nodes) != 1 and len(bin_matmul_nodes) != 2:
        print(f"Expect 2 bmm/matmul op in the {org_class_name}, found {len(bmm_nodes)}")
        print(f"Or expect 1 sdpa op in the {org_class_name}, found {len(sdpa_nodes)}")
        print(f"Or expect 2 @ op in the {org_class_name}, found {len(bin_matmul_nodes)}")
        print("Auto quantization of KV Cache fails")
        return False

    if len(bmm_nodes) == 2:
        patch(bmm_nodes[0], quantizer_names=(None, "k_bmm_quantizer"))
        patch(bmm_nodes[1], quantizer_names=(None, "v_bmm_quantizer"))
        print("Patching 2 BMM/Matmul operators with quantizers")
    if len(bin_matmul_nodes) == 2:
        patch_binop(bin_matmul_nodes[0], quantizer_names=(None, "k_bmm_quantizer"))
        patch_binop(bin_matmul_nodes[1], quantizer_names=(None, "v_bmm_quantizer"))
        print("Patching 2 @ operators with quantizers")

    if len(sdpa_nodes) == 1:
        patch(sdpa_nodes[0], quantizer_names=(None, "k_bmm_quantizer", "v_bmm_quantizer"))
        print("Patching 1 scaled_dot_product_attention operator with quantizers")

    head = ast.fix_missing_locations(head)
    org_class = model_module.__dict__[org_class_name]

    module_code_str = ast.unparse(head)
    with tempfile.NamedTemporaryFile(prefix="modelopt_", suffix=".py", delete=False) as temp_file:
        temp_file.write(module_code_str.encode())
        print(f"Definition of {new_class_name} saved to {temp_file.name}")

    # Exec with python runtime and extract the new class
    # This could lead to side effects if the class code is not properly isolated
    # Therefore, it is recommended to run this function only when necessary
    # exec(
    #     new_class_code,
    #     globals=model_module.__dict__,
    #     locals=model_module.__dict__
    # )  # bandit throws error here
    # quant_class = model_module.__dict__[new_class_name]

    # Extract the bytecode and create a new class on the fly
    # This is more tricky but doesn't require runtime execution
    module_code = compile(head, filename="modelopt_generated", mode="exec")
    class_code = module_code.co_consts[0]
    assert class_code.co_name == new_class_name
    method_codes = [const for const in class_code.co_consts if isinstance(const, types.CodeType)]

    new_methods = {}
    for method_code in method_codes:
        method_name = method_code.co_name
        original_method = getattr(org_class, method_name, None)
        if not isinstance(original_method, types.FunctionType):
            continue
        # Create a new class method from bytecode
        new_methods[method_name] = types.FunctionType(
            method_code, globals=original_method.__globals__, closure=original_method.__closure__
        )

    def setup_method(self):
        self.k_bmm_quantizer = TensorQuantizer()
        self.v_bmm_quantizer = TensorQuantizer()

    assert "_setup" not in new_methods, "Method _setup already exists"
    new_methods["_setup"] = setup_method

    # Create a new subclass on the fly
    quant_class = type(new_class_name, (org_class,), new_methods)

    mtq.register(original_cls=org_class, quantized_cls=quant_class)
    print(f"Successfully registered {org_class_name} for quantization")
    return True
