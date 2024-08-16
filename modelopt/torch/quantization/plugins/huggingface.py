# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Support quantization for huggingface layers."""
import torch
import torch.nn as nn
import transformers

from modelopt.torch.opt.dynamic import DynamicModule

from ..nn import QuantModuleRegistry
from ..nn.modules.quant_linear import _QuantLinear
from .attention import register_attention_for_kv_quant

__all__ = ["register_hf_attentions_on_the_fly"]

if transformers.modeling_utils.Conv1D not in QuantModuleRegistry:
    # transformers.modeling_utils.Conv1D used in HF-GPT2 is not a real Conv1D
    # It is actually a Linear layer where weight is transposed and torch.addmm is used
    @QuantModuleRegistry.register({transformers.modeling_utils.Conv1D: "Conv1D"})
    class _QuantConv1D(_QuantLinear):
        @classmethod
        @torch.no_grad()
        def convert(cls, module: nn.Module) -> "_QuantConv1D":
            module.weight = nn.Parameter(module.weight.T.contiguous())
            module.out_features, module.in_features = module.weight.shape
            # We want the forward method of nn.Linear to be called instead of the forward method of Conv1D
            dyn_cls: DynamicModule = QuantModuleRegistry.get(nn.Linear)
            return dyn_cls.convert(module)


if hasattr(transformers.models, "dbrx"):
    # For more information on DbrxExpert, see https://github.com/huggingface/transformers/blame/dcdda5324bcc7a750b5e40e11dd795442204ff27/src/transformers/models/dbrx/modeling_dbrx.py#L756
    class _QuantDbrxExperts(DynamicModule):
        def _setup(self):
            """Modify the DbrxExpert."""
            # No setup is needed for DbrxExpert, we only need to update DbrxExpertGLU
            pass

        # forward method copied from the original dbrx repo - https://github.com/databricks/dbrx/blob/a3200393e678387a6f30f3e903108c650625eb21/model/modeling_dbrx.py#L795
        def forward(
            self,
            x: torch.Tensor,
            weights: torch.Tensor,
            top_weights: torch.Tensor,
            top_experts: torch.LongTensor,
        ) -> torch.Tensor:
            bsz, q_len, hidden_size = x.shape
            x = x.view(-1, hidden_size)
            out = torch.zeros_like(x)

            expert_mask = nn.functional.one_hot(
                top_experts, num_classes=self.moe_num_experts
            ).permute(2, 1, 0)
            for expert_idx in range(0, self.moe_num_experts):
                topk_idx, token_idx = torch.where(expert_mask[expert_idx])
                if token_idx.shape[0] == 0:
                    continue

                token_list = token_idx.tolist()
                topk_list = topk_idx.tolist()

                expert_tokens = x[None, token_list].reshape(-1, hidden_size)
                expert_out = (
                    self.mlp(expert_tokens, expert_idx) * top_weights[token_list, topk_list, None]
                )

                out.index_add_(0, token_idx, expert_out)

            out = out.reshape(bsz, q_len, hidden_size)
            return out

    class _QuantDbrxExpertGLU(DynamicModule):

        def _setup(self):
            """Modify the DbrxExpertGLU by using nn.Linear layers."""
            dtype, device = self.w1.dtype, self.w1.device

            def _copy_weights(modules, weights):
                modules.to(dtype=dtype, device=device)
                for expert_idx, module in enumerate(modules):
                    with torch.no_grad():
                        module.weight.copy_(weights[expert_idx].detach())

            self.w1_linear = nn.ModuleList(
                [
                    nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=False)
                    for _ in range(self.moe_num_experts)
                ]
            )
            _copy_weights(
                self.w1_linear,
                self.w1.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size),
            )
            delattr(self, "w1")

            self.v1_linear = nn.ModuleList(
                [
                    nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=False)
                    for _ in range(self.moe_num_experts)
                ]
            )
            _copy_weights(
                self.v1_linear,
                self.v1.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size),
            )
            delattr(self, "v1")

            self.w2_linear = nn.ModuleList(
                [
                    nn.Linear(self.ffn_hidden_size, self.hidden_size, bias=False)
                    for _ in range(self.moe_num_experts)
                ]
            )
            _copy_weights(
                self.w2_linear,
                self.w2.view(
                    self.moe_num_experts, self.ffn_hidden_size, self.hidden_size
                ).transpose(1, 2),
            )
            delattr(self, "w2")

        def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
            x1 = self.w1_linear[expert_idx](x)
            x2 = self.v1_linear[expert_idx](x)
            x1 = self.activation_fn(x1)
            x1 = x1 * x2
            return self.w2_linear[expert_idx](x1)

    if transformers.models.dbrx.modeling_dbrx.DbrxExperts not in QuantModuleRegistry:
        QuantModuleRegistry.register(
            {transformers.models.dbrx.modeling_dbrx.DbrxExperts: "hf.DbrxExperts"}
        )(_QuantDbrxExperts)

    if transformers.models.dbrx.modeling_dbrx.DbrxExpertGLU not in QuantModuleRegistry:
        QuantModuleRegistry.register(
            {transformers.models.dbrx.modeling_dbrx.DbrxExpertGLU: "hf.DbrxExpertGLU"}
        )(_QuantDbrxExpertGLU)


if hasattr(transformers.models, "falcon") and hasattr(
    transformers.models.falcon.modeling_falcon, "FalconLinear"
):
    QuantModuleRegistry.register(
        {transformers.models.falcon.modeling_falcon.FalconLinear: "FalconLinear"}
    )(_QuantLinear)


def register_dbrx_moe_on_the_fly(model):
    """Register DBRX MoE modules as QUANT_MODULE.

    The MoE class in DBRX is `transformers_modules.modeling_dbrx.DbrxExpertGLU`, which loads dynamically.
    """
    if type(model).__name__ in ["DbrxForCausalLM"]:
        moe_type = type(model.transformer.blocks[0].ffn.experts.mlp)
        # Create a QuantFalconLinear class on the fly
        if QuantModuleRegistry.get(moe_type) is None:
            QuantModuleRegistry.register({moe_type: moe_type.__name__})(_QuantDbrxExpertGLU)


def register_falcon_linears_on_the_fly(model):
    """Register Falcon linear modules as a QUANT_MODULE.

    Certain falcon models (for example, falcon 40b) use remote code, which are loaded dynamically, to build their model.
    Therefore, we need to register the linear on the fly before quantization.
    """
    if type(model).__name__ in ["RWForCausalLM", "FalconForCausalLM"]:
        linear_type = type(model.transformer.h[0].self_attention.dense)
        # Create a QuantFalconLinear class on the fly
        if QuantModuleRegistry.get(linear_type) is None:
            QuantModuleRegistry.register({linear_type: linear_type.__name__})(_QuantLinear)


def register_hf_attentions_on_the_fly(model):
    """Find HF Attention modules in the model and register them for KV Cache quantization.

    This function attempts to find child modules ending with "Attention" in the name.
    If such child modules are not found, or the corresponding class does not contain
    identifiable attention patterns, the function will not register any new modules.
    """
    attention_cls = {}
    for name, module in model.named_modules():
        if type(module).__name__.endswith("Attention"):
            attention_cls[type(module)] = type(module).__name__

    success = any([register_attention_for_kv_quant(cls) for cls in attention_cls])
    if not success:
        raise RuntimeError("No Attention class found for KV Cache quantization.")
