# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Custom plugin base modules and utilities for quantization."""

from contextlib import ExitStack, contextmanager
from functools import partial
from types import ModuleType
from typing import Callable, Iterator, List, Tuple

from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.nn.modules.quant_linear import _QuantLinear

from ..utils import replace_function

try:
    from .huggingface import register_dbrx_moe_on_the_fly, register_falcon_linears_on_the_fly
except ImportError:

    def _dummy_register(model):
        pass

    register_falcon_linears_on_the_fly = _dummy_register
    register_dbrx_moe_on_the_fly = _dummy_register


# TODO: This is a temporary solution
# In future implement a decorator to register methods updating QUANT_MODULE on the fly
def register_custom_model_plugins_on_the_fly(model):
    """Registers custom modules as QUANT_MODULE on the fly."""
    register_falcon_linears_on_the_fly(model)
    register_dbrx_moe_on_the_fly(model)


@contextmanager
def _multi_context(*cms):
    """Context manager enabling variable number of context managers."""
    with ExitStack() as stack:
        yield [stack.enter_context(cls) for cls in cms]


class _QuantFunctionalMixin(DynamicModule):
    """Mixin class for quantized functionals.

    Often we need to replace a functional with a quantized version. This class provides a way to do that.
    """

    # List of functionals to replace with quantized versions, e.g. [(package, func_name, quantized_func), ...]
    _functionals_to_replace: List[Tuple[ModuleType, str, Callable]] = []

    @property
    def functionals_to_replace(self) -> Iterator[Tuple[ModuleType, str, Callable]]:
        return (
            (package, func_name, quantized_func)
            for package, func_name, quantized_func in self._functionals_to_replace
            if hasattr(package, func_name)
        )

    def forward(self, *args, **kwargs):
        with _multi_context(
            *(
                replace_function(package, func_name, quantized_func)
                for package, func_name, quantized_func in self.functionals_to_replace
            )
        ):
            return super().forward(*args, **kwargs)


class _ParallelLinear(_QuantFunctionalMixin):
    """Quantized base class for ParallelLinear type classes.

    For this type of modules, we need to quantize the inputs and weights just before calling the actual linear
    functional. This is accomplished by replacing the linear functional with a custom one that quantizes the inputs
    and weights before calling the original functional.
    """

    # List of functionals to replace [(package, func_name), ...]
    _functionals_to_replace: List[Tuple[ModuleType, str]] = []

    @property
    def functionals_to_replace(self) -> Iterator[Tuple[ModuleType, str, Callable]]:
        for package, func_name in self._functionals_to_replace:
            if not hasattr(package, func_name):
                continue
            quantized_func = partial(
                _QuantLinear.quantized_linear_fn, package, "_" + func_name, self
            )
            if hasattr(getattr(package, func_name), "__dict__"):
                quantized_func.__dict__.update(getattr(package, func_name).__dict__)
            yield package, func_name, quantized_func

    def _setup(self):
        self.input_quantizer = TensorQuantizer(_QuantLinear.default_quant_desc_input)
        self.weight_quantizer = TensorQuantizer(_QuantLinear.default_quant_desc_weight)
        self.output_quantizer = TensorQuantizer(_QuantLinear.default_quant_desc_output)
        self.output_quantizer.disable()
