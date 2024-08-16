# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Module for advanced quantization algorithms."""
import gc
import types
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import regex as re
import torch
import torch.nn as nn

from modelopt.torch.opt.hparam import CustomHPType, Hparam, HPType
from modelopt.torch.opt.searcher import LPS, BaseSearcher, SearchConfig, SearchStateDict
from modelopt.torch.opt.utils import get_hparam, named_hparams
from modelopt.torch.utils import create_param_grad_clear_hook, print_rank_0, report_memory

from . import config as mtq_config
from .conversion import set_quantizer_by_cfg
from .model_calib import calibrate
from .nn import QuantLinearConvBase, TensorQuantizer
from .utils import is_quantized_linear


class QuantRecipe(CustomHPType):
    """A subclass of QuantizeConfig enabling auto_quantize specific configurations."""

    UNSUPPORTED_RECIPES = ["AWQ", "SMOOTHQUANT"]

    def __init__(self, name: Optional[str] = None):
        """Initialize the QuantRecipe with the name of the quantization format."""
        assert name in mtq_config.choices or name is None
        if name:
            assert all(
                n not in name for n in QuantRecipe.UNSUPPORTED_RECIPES
            ), f"Unsupported quantization format {name}"
        self.name = name

    @property
    def config(self) -> mtq_config.QuantizeConfig:
        """Get the quantization configuration for the quantization format."""
        if self.name is None:
            cfg = mtq_config.QuantizeConfig(quant_cfg={"*": {"enable": False}}, algorithm="max")
        else:
            cfg = mtq_config.QuantizeConfig(**getattr(mtq_config, self.name))
            # Disable KV Cache quantization
            # Currently KV Cache quantization is enabled for some quantization formats and disabled for others
            # This breaks the monotonicity of the quantization formats in terms of weight compression Vs accuracy
            cfg.quant_cfg["*output_quantizer"] = mtq_config.QuantizerAttributeConfig(enable=False)
        return cfg

    @property
    def compression(self) -> float:
        """Get the compression factor for the quantization format."""
        if self.name is None:
            return 1.0
        if self.name.split("_")[0] in ["INT8", "FP8"]:
            return 0.5
        elif self.name.split("_")[0] in ["INT4", "W4A8"]:
            return 0.25
        else:
            raise ValueError(f"Unsupported quantization format {self.name}")

    def __repr__(self) -> str:
        return f"{self.name}"

    def __lt__(self, other: "QuantRecipe"):
        return self.compression < other.compression

    def __eq__(self, other: object):
        assert isinstance(other, QuantRecipe)
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


class QuantRecipeHparam(Hparam):
    """An Hparam for quantization recipes.

    In addition, this Hparam also:
    1. Keeps a link to its modules and sets the quantizers for the module based on the active recipe.
    2. Keeps track of the importance of each recipe in a dict instead of a tensor
    3. Links to other QuantRecipeHparam objects to enable setting the same recipe for multiple modules.
    """

    def __init__(
        self,
        choices: Sequence[QuantRecipe],
        original: Optional[QuantRecipe] = None,
        nn_module: nn.Module = None,
    ) -> None:
        """Initializes Hparam with original value and choices."""
        choices = sorted(set(choices) | {QuantRecipe(None)})
        super().__init__(choices, original)
        self.nn_module = nn_module

        self._parent: QuantRecipeHparam = self
        self._children: List[QuantRecipeHparam] = []

        # This is a hack; We dont want to make the input_quantizer, weight_quantizer, output_quantizer
        # a dynamic attribute for backward compatibility with the model_calib.py
        # TODO: Make input_quantizer, weight_quantizer, output_quantizer a dynamic attribute and get rid of this hack
        self._quantizer_choices = {
            quant_recipe: {
                "input_quantizer": TensorQuantizer(),
                "weight_quantizer": TensorQuantizer(),
                "output_quantizer": TensorQuantizer(),
            }
            for quant_recipe in self.choices
        }
        quant_recipe: QuantRecipe
        for quant_recipe in self.choices:
            self.active = quant_recipe
            if self.nn_module is None:
                continue
            set_quantizer_by_cfg(self.nn_module, quant_recipe.config.quant_cfg)

        self.active = self.original

        self._importance_dict = {quant_recipe: 0.0 for quant_recipe in self.choices}

    @property
    def active(self) -> HPType:
        """Return the currently active value."""
        return self._active

    @active.setter
    def active(self, val: Optional[HPType]):
        """Set the active value with a sanity check for choices and dynamic hparams."""
        val = self.original if val is None else val
        assert val in self._choices, f"val = {val}, choices = {self.choices}"
        if self.is_configurable:
            self._active = val
        else:
            assert self._active == val
        if self.nn_module is None:
            return
        for quantizer_attr_name, quantizer in self._quantizer_choices[val].items():
            setattr(self.nn_module, quantizer_attr_name, quantizer)

        for child_hp in self._children:
            with child_hp._force_configurable():
                child_hp.active = val

    # TODO: This should be handled by Symbol
    def link_to(self, other: "QuantRecipeHparam") -> None:
        """Link this QuantRecipeHparam to the other QuantRecipeHparam."""
        assert (
            self.is_configurable and other.is_configurable
        ), "Both hparams must be configurable to link"
        assert self != other, "Cannot link to self"

        for child in self._children + [self]:
            child._parent = other
            other._children.append(child)

        self._children = []
        self._is_configurable = False


class AutoQuantizeSearcher(BaseSearcher):
    """A searcher for AutoQuantize algorithm.

    In AutoQuantize, we search for the best per-layer quantization configuration that minimizes the sum of per-layer
    scores while meeting the specified constraint. AutoQuantize uses Linear Programming Solver to find the
    optimal quantization configuration.

    The auto_quantize score for a layer quantization configuration is an approximation of model loss change change due
    to quantizing the particular layer with the particular configuration.
    The approximation is based on taylor expansion of the loss function wrt to the quantized output of the layer and
    substitution of Fisher information for Hessian.
    This approximation is mathematically correct for models where the loss
    is a log likelihood loss such as BERT, GPT, etc. However, the auto_quantize score can still be used as a proxy
    for other models such as ResNet.
    """

    candidate_stats: Dict[str, Dict[str, List[float]]]

    rules = [
        r"^(.*?)\.(q_proj|k_proj|v_proj)$",  # q_proj, k_proj, v_proj for llama like models
        r"^(.*?)\.(gate_proj|up_proj)$",  # gate_proj, up_proj for llama like models
        r"^(.*?)\.(\d+\.(w1|w2|w3))$",  # mixtral experts
        r"^(.*?)\.((w1_linear|w2_linear|w3_linear)\.\d+)$",  # dbrx experts
    ]

    @property
    def default_search_config(self):
        """Get the default search config for AutoQuantize."""
        config_dict = super().default_search_config
        config_dict.pop("max_iter_data_loader")
        config_dict.update(
            {
                "quantization_formats": ["FP8_DEFAULT_CFG", None],
                "num_calib_steps": 512,
                "num_score_steps": 128,
            }
        )
        return config_dict

    @property
    def default_state_dict(self) -> SearchStateDict:
        """Get the default state dict for AutoQuantize."""
        return {
            "candidate_stats": defaultdict(dict),
            "best": {"recipe": {}, "constraints": {}, "score": float("inf"), "is_satisfied": False},
        }

    def sanitize_search_config(self, config: Optional[SearchConfig]) -> SearchConfig:
        """Sanitize the search config dict."""
        config = config or {}
        if "score_func" in config:
            warnings.warn("`score_func` is ignored for `auto_quantize`.")
        config["score_func"] = lambda x: 0.0
        config = super().sanitize_search_config(config)
        config.pop("score_func")
        assert (
            config["data_loader"] is not None
        ), "data_loader must be provided for `auto_quantize`."
        return config

    @staticmethod
    def _is_auto_quantize_module(module):
        return is_quantized_linear(module) or isinstance(module, QuantLinearConvBase)

    @staticmethod
    def _get_search_recipes(quantization_formats):
        return sorted(set([QuantRecipe(name=format) for format in quantization_formats]))

    @torch.enable_grad()
    def _estimate_auto_quantize_scores(self):
        # TODO: remove the no-quant recipe
        def auto_quantize_score_estimate_forward(module, input, *args, **kwargs):
            module.quant_recipe = QuantRecipe(None)
            output = module._forward_original(input, *args, **kwargs)

            # If gradient checkpointing is enabled, gradient will not be enabled in the global forward pass.
            # With gradient checkpointing, gradients are computed in the local forward pass during backward pass

            # Lets compute the output_diff and save it in memory only if gradient is enabled to be memory efficient
            if not torch.is_grad_enabled():
                return output

            module.output_diff_dict = {}
            with torch.no_grad():
                for recipe in module.get_hparam("quant_recipe").choices:
                    if recipe.name is None:
                        continue
                    module.quant_recipe = recipe
                    output_diff = module._forward_original(input, *args, **kwargs)

                    if isinstance(output_diff, tuple):
                        output_diff = output_diff[0] - output[0]
                    else:
                        output_diff -= output
                    module.output_diff_dict[recipe] = output_diff

            return output

        def backward_hook(module, grad_input, grad_output):
            for recipe, output_diff in module.output_diff_dict.items():
                score = ((grad_output[0].float() ** 2) * (output_diff.float() ** 2)).sum()
                module.get_hparam("quant_recipe")._importance_dict[recipe] += score.item()

            del module.output_diff_dict
            gc.collect()

        def setup_params_for_score_estimation(name, param, params_metadata):
            params_metadata[name] = {"requires_grad": param.requires_grad}
            param.requires_grad = True
            accum_grad, handle = create_param_grad_clear_hook(param)
            params_metadata[name]["accum_grad"] = accum_grad  # We need to keep the accum_grad alive
            params_metadata[name]["handle"] = handle

        def setup_module_for_score_estimation(module):
            module._forward_original = module.forward
            module.forward = types.MethodType(auto_quantize_score_estimate_forward, module)
            module._backward_hook_handle = module.register_full_backward_hook(backward_hook)

        def cleanup_module_after_score_estimation(module):
            module.forward = module._forward_original
            del module._forward_original

            module._backward_hook_handle.remove()

        def cleanup_params_after_score_estimation(name, param, params_metadata):
            param.requires_grad = params_metadata[name]["requires_grad"]
            params_metadata[name]["handle"].remove()

        for name, module in self.model.named_modules():
            if self._is_auto_quantize_module(module):
                # Monkey patch the forward methods to cache Y(Q(W), Q(X)) - Y(W,X)
                setup_module_for_score_estimation(module)

        params_metadata = {}
        for name, param in self.model.named_parameters():
            # Let us delete the gradient as soon as they are computed to save memory
            # In addition, this method enables gradient for all parameters
            # This is needed to make sure the re-entrant activation checkpointing works
            setup_params_for_score_estimation(name, param, params_metadata)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            report_memory("AutoQuantize: starting score estimation, ")

        def run_backward(output, batch):
            loss = self.config["loss_func"](output, batch)
            loss.backward()

        forward_loop = self.construct_forward_loop(
            silent=False,
            progress_bar_msg="Estimating auto_quantize scores",
            post_process_fn=run_backward,
            max_iter_data_loader=self.config["num_score_steps"],
        )

        forward_loop(self.model)  # type: ignore[misc]

        if torch.cuda.is_available():
            report_memory("AutoQuantize: After score estimation")

        for name, module in self.model.named_modules():
            if self._is_auto_quantize_module(module):
                cleanup_module_after_score_estimation(module)

        for name, param in self.model.named_parameters():
            cleanup_params_after_score_estimation(name, param, params_metadata)

        # Delete the params_metadata
        del params_metadata
        gc.collect()

    @classmethod
    def insert_quant_recipe_hparams(cls, model: nn.Module, quant_recipes: List[QuantRecipe]):
        """Insert the QuantRecipeHparam into the model for each quantized module."""
        for name, module in model.named_modules():
            if cls._is_auto_quantize_module(module):
                hparam = QuantRecipeHparam(
                    quant_recipes,
                    original=quant_recipes[0],
                    nn_module=module,
                )
                module._register_hparam("quant_recipe", hparam)

    def before_search(self):
        """Prepare the model for search by calibrating the quantizers  and collecting ``AutoQuantize`` score."""
        search_recipes = self._get_search_recipes(self.config["quantization_formats"])
        self.insert_quant_recipe_hparams(self.model, search_recipes)

        # Iterate over the search recipes and calibrate the quantizers for each recipe
        for recipe in search_recipes:
            if recipe.name is None:
                continue
            for name, module in self.model.named_modules():
                if self._is_auto_quantize_module(module):
                    # Set the quantizers for the recipe
                    module.quant_recipe = recipe

            # Now calibrate the quantizers for the recipe
            calibrate(
                self.model,
                algorithm=recipe.config.algorithm,
                forward_loop=self.construct_forward_loop(
                    silent=False,
                    progress_bar_msg=f"Calibrating for {recipe.name}",
                    max_iter_data_loader=self.config["num_calib_steps"],
                ),
            )

        self.model.eval()
        # Huggingface transformers: Enable gradient checkpointing to save memory during backward pass
        if hasattr(self.model, "gradient_checkpointing_enable"):
            print_rank_0(
                "AutoQuantize: Huggingface model detected - Enabling gradient checkpointing "
            )
            self.model.gradient_checkpointing_enable({"use_reentrant": True})
            self.model.train()

        self._estimate_auto_quantize_scores()

    @classmethod
    def merge_search_hparam_by_rules(cls, model):
        """Restrict the search space so that multiple modules can share the same recipe."""
        # TRTLLM fuses linear layers such as q_proj, k_proj, v_proj into same layer
        # Hence we need to restrict the search space so that all these layers share the same recipe

        prefix_to_hparam_map: Dict[str, QuantRecipeHparam] = {}
        for name, module in model.named_modules():
            if not cls._is_auto_quantize_module(module):
                continue
            for rule in cls.rules:
                pattern = re.compile(rule)
                match = pattern.match(name)
                if match:
                    hparam: QuantRecipeHparam = module.get_hparam("quant_recipe")
                    prefix = match.group(1)
                    if prefix not in prefix_to_hparam_map:
                        prefix_to_hparam_map[prefix] = hparam
                    else:
                        hparam.link_to(prefix_to_hparam_map[prefix])

    def run_search(self):
        """Search for the best per-layer quantization configuration and return the best model and configuration.

        AutoQuantize uses Linear Programming Solver to find the optimal quantization configuration which
        minimizes the sum of per-layer auto_quantize scores while meeting the specified constraint.
        """

        def get_total_weight_size(modules):
            return sum(
                map(
                    lambda module: (
                        module.weight.numel() if self._is_auto_quantize_module(module) else 0
                    ),
                    modules,
                )
            )

        def _get_constraints_for_search(lower_bound=None):
            total_model_weight_size = get_total_weight_size(self.model.modules())

            upper_bound = self.constraints["weight_compression"]
            if isinstance(upper_bound, str):
                assert upper_bound.endswith(
                    "%"
                ), f"Unsupported format for weight_compression constraint {upper_bound}"
                upper_bound = float(upper_bound[:-1]) / 100

            if lower_bound:
                lower_bound = lower_bound * upper_bound

            constraints = {
                "weight_size_after_compression": (
                    lower_bound * total_model_weight_size if lower_bound else lower_bound,
                    upper_bound * total_model_weight_size,
                )
            }
            return constraints, "weight_size_after_compression"

        verbose = self.config["verbose"]
        assert (
            len(self.constraints) == 1 and "weight_compression" in self.constraints
        ), f"`constraints` must contain only 'weight_compression' constraint. Got {self.constraints.keys()}"

        self.merge_search_hparam_by_rules(self.model)

        search_recipes = self._get_search_recipes(self.config["quantization_formats"])
        for name, hparam in named_hparams(self.model, configurable=True):
            formats, scores, costs = [], [], []
            prev_score = float("inf")
            for recipe in search_recipes:
                formats.append(recipe.name)
                score = hparam._importance_dict[recipe]  # type: ignore[attr-defined]
                for child_hp in hparam._children:  # type: ignore[attr-defined]
                    score += child_hp._importance_dict[recipe]
                scores.append(min(score, prev_score))
                costs.append(
                    get_total_weight_size(
                        [hparam.nn_module] + [child_hp.nn_module for child_hp in hparam._children]  # type: ignore[attr-defined]
                    )
                    * recipe.compression
                )
                prev_score = score
            self.candidate_stats[name]["formats"] = formats
            self.candidate_stats[name]["scores"] = scores
            self.candidate_stats[name]["costs"] = costs

        for lower_bound in [None, 0.99, 0.90]:
            # The LP solver for auto_quantize sometimes fails to find a solution if a lower bound is not
            # specified. I dont know why this happens.
            # As a workaround, lets specify a lower bound for the weight compression if previous
            # search without lower bound fails.
            constraints, constraint_name = _get_constraints_for_search(lower_bound)

            lps = LPS(
                name="AutoQuantize",
                constraints=constraints,
                constraints_to_candidate_costs={
                    constraint_name: [
                        candidate_stat["costs"] for candidate_stat in self.candidate_stats.values()
                    ]
                },
                candidate_scores=[
                    candidate_stat["scores"] for candidate_stat in self.candidate_stats.values()
                ],
                objective_type="minimize",
                verbose=verbose,
            )
            selections, self.status = lps()
            if self.status == "Optimal":
                break

        self.best = {}
        if self.status != "Optimal":
            warnings.warn(
                "AutoQuantize FAILED to find a solution! The searched model might not meet all constraints. "
            )

        best_recipe = {}
        best_constraints, best_scores = 0, 0
        for name, selected_idx in zip(self.candidate_stats.keys(), selections):
            best_recipe[name] = self.candidate_stats[name]["formats"][selected_idx]
            hparam = get_hparam(self.model, name)
            hparam.active = QuantRecipe(best_recipe[name])  # type: ignore[arg-type]
            best_constraints += self.candidate_stats[name]["costs"][selected_idx]
            best_scores += self.candidate_stats[name]["scores"][selected_idx]
            if verbose:
                print_rank_0(
                    f"AutoQuantize best recipe for {name.replace('.quant_recipe', '')}: {best_recipe[name]}"
                )

        self.best["recipe"] = best_recipe
        self.best["constraints"] = {constraint_name: best_constraints}
        self.best["score"] = best_scores
        self.best["is_satisfied"] = best_constraints <= constraints[constraint_name][1]
