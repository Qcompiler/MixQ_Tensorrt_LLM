# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""This document lists the quantization formats supported by Model Optimizer and example quantization configs.

.. _quantization-formats:

Quantization Formats
==========================================

The following table lists the quantization formats supported by Model Optimizer and the corresponding quantization
config. See :ref:`Quantization Configs <example-quantization-configs>` for the
specific quantization config definitions.

Please see :doc:`choosing the right quantization formats <../../guides/_choosing_quant_methods>` to
learn more about the formats and their use-cases.

.. note::

    The recommended configs given below are for LLM models. For CNN models, only INT8 quantization
    is supported. Please use quantization config ``INT8_DEFAULT_CFG`` for CNN models.

=================================   =======================================================
Quantization  Format                Model Optimizer config
=================================   =======================================================
INT8                                ``INT8_SMOOTHQUANT_CFG``

FP8                                 ``FP8_DEFAULT_CFG``

INT4 Weights only AWQ (W4A16)       ``INT4_AWQ_CFG``

INT4-FP8 AWQ (W4A8)                 ``W4A8_AWQ_BETA_CFG``

=================================   =======================================================

.. _quantization-configs:

Quantization Configs
================================

Quantization config is dictionary specifying the values for keys ``"quant_cfg"`` and
``"algorithm"``. The ``"quant_cfg"`` key specifies the quantization configurations. The
``"algorithm"`` key specifies the ``algorithm`` argument to
:meth:`calibrate <modelopt.torch.quantization.model_calib.calibrate>`. Please see :class:`QuantizeConfig`
for the quantization config definition.

'Quantization configurations' is a dictionary mapping wildcards or filter functions
to its 'quantizer attributes'. The wildcards or filter functions  are matched
against the quantizer module names. The quantizer modules have names ending with
``weight_quantizer`` and ``input_quantizer`` and they perform weight quantization and
input quantization (or activation quantization) respectively. The quantizer modules are generally
instances of
:class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer>`.
The quantizer attributes are defined by :class:`QuantizerAttributeConfig`. See :class:`QuantizerAttributeConfig`
for details on the quantizer attributes and their values.

The key `"default"` from the quantization configuration dictionary is applied if no other wildcard or filter functions
match the quantizer module name.

The quantizer attributes are applied in the order they are specified. For the missing attributes, the default attributes
as defined by :class:`QuantizerAttributeConfig` are used.

Quantizer attributes can also be a list of dictionaries. In this case, the matched quantizer module
is replaced with a
:class:`SequentialQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.SequentialQuantizer>`
module which is used to quantize a tensor in multiple formats sequentially. Each quantizer attribute
dictionary in the list specifies the quantization formats for each quantization step of the
sequential quantizer. For example, `SequentialQuantizer` is used in 'INT4 Weights, FP8 Activations'
quantization in which the weights are quantized in INT4 followed by FP8.

In addition, the dictionary entries could also be pytorch module class names mapping the class specific
quantization configurations. The pytorch modules should have a quantized equivalent.

To get the string representation of a module class, do:

.. code-block::

    from modelopt.torch.quantization.nn import QuantModuleRegistry

    # Get the class name for nn.Conv2d
    class_name = QuantModuleRegistry.get_key(nn.Conv2d)

Here is an example of a quantization config:

.. code-block::

    MY_QUANT_CFG = {
        "quant_cfg": {
            # Quantizer wildcard strings mapping to quantizer attributes
            "*weight_quantizer": {"num_bits": 8, "axis": 0},
            "*input_quantizer": {"num_bits": 8, "axis": None},

            # Module class names mapping to quantizer configurations
            "nn.LeakyReLU": {"*input_quantizer": {"enable": False}},

        }
    }

.. _example-quantization-configs:

Example Quantization Configurations
==========================================

Here are the recommended quantization configs from Model Optimizer for
quantization formats such as FP8, INT8, INT4, etc.:

.. code-block::

    INT8_DEFAULT_CFG = {
        "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": None},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*router*": {"enable": False},  # Skip the MOE router
        "default": {"enable": False},
        },
        "algorithm": "max",
    }

    INT8_SMOOTHQUANT_CFG = {
        "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": -1},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*router*": {"enable": False},  # Skip the MOE router
        "nn.Conv2d": {
            "*weight_quantizer": {"num_bits": 8, "axis": 0},
            "*input_quantizer": {"num_bits": 8, "axis": None},
        },
        "default": {"enable": False},
        },
        "algorithm": "smoothquant",
    }

    FP8_DEFAULT_CFG = {
        "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*router*": {"enable": False},  # Skip the MOE router
        "default": {"enable": False},
        },
        "algorithm": "max",
    }

    INT4_BLOCKWISE_WEIGHT_ONLY_CFG = {
        "quant_cfg": {
        "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
        "*input_quantizer": {"enable": False},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*router*": {"enable": False},  # Skip the MOE router
        "default": {"enable": False},
        },
        "algorithm": "max",
    }

    INT4_AWQ_CFG = {
        "quant_cfg": {
        "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
        "*input_quantizer": {"enable": False},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*router*": {"enable": False},  # Skip the MOE router
        "default": {"enable": False},
        },
        "algorithm": {"method": "awq_lite", "alpha_step": 0.1},
        # "algorithm": {"method": "awq_full", "alpha_step": 0.1, "max_co_batch_size": 1024},
        # "algorithm": {"method": "awq_clip", "max_co_batch_size": 2048},
    }

    W4A8_AWQ_BETA_CFG = {
    "quant_cfg": {
        "*weight_quantizer": [
            {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
            {"num_bits": (4, 3), "axis": None, "enable": True},
        ],
        "*input_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*router*": {"enable": False},  # Skip the MOE router
        "default": {"enable": False},
    },
    "algorithm": "awq_lite",
    }

These config can be accessed as attributes of ``modelopt.torch.quantization`` and can be given as
input to :meth:`mtq.quantize() <modelopt.torch.quantization.model_quant.quantize>`. For example:

.. code-block::

    import modelopt.torch.quantization as mtq
    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)

You can also create your own config by following these examples.
For instance, if you want to quantize a model with int4 AWQ algorithm, but need to skip quantizing
the layer named ``lm_head``,  you can create a custom config and quantize your model as following:

.. code-block::

    # Create custom config
    CUSTOM_INT4_AWQ_CFG = copy.deepcopy(mtq.INT4_AWQ_CFG)
    CUSTOM_INT4_AWQ_CFG["quant_cfg"]["*lm_head*"] = {"enable": False}

    # quantize model
    model = mtq.quantize(model, CUSTOM_INT4_AWQ_CFG, forward_loop)

"""
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import ValidationInfo, field_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField
from modelopt.torch.utils.network import ConstructorLike

_default_disabled_quantizer_cfg = {
    "nn.BatchNorm1d": {"*": {"enable": False}},
    "nn.BatchNorm2d": {"*": {"enable": False}},
    "nn.BatchNorm3d": {"*": {"enable": False}},
    "nn.LeakyReLU": {"*": {"enable": False}},
}

INT8_DEFAULT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": None},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*router*": {"enable": False},  # Skip the MOE router
        "*output_layer*": {"enable": False},
        **_default_disabled_quantizer_cfg,
        "default": {"enable": False},
    },
    "algorithm": "max",
}

INT8_MIX = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (8, 16), "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": -1},
        "*lm_head*": {"enable": False},
        "*output_layer*": {"enable": False},
        "default": {"num_bits": 8, "axis": None},
    },
    "algorithm": "max",
}

INT8_SMOOTHQUANT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": -1},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*router*": {"enable": False},  # Skip the MOE router
        "*output_layer*": {"enable": False},
        "nn.Conv1d": {
            "*weight_quantizer": {"num_bits": 8, "axis": 0},
            "*input_quantizer": {"num_bits": 8, "axis": None},
        },
        "nn.Conv2d": {
            "*weight_quantizer": {"num_bits": 8, "axis": 0},
            "*input_quantizer": {"num_bits": 8, "axis": None},
        },
        "nn.Conv3d": {
            "*weight_quantizer": {"num_bits": 8, "axis": 0},
            "*input_quantizer": {"num_bits": 8, "axis": None},
        },
        **_default_disabled_quantizer_cfg,
        "default": {"enable": False},
    },
    "algorithm": "smoothquant",
}

FP8_DEFAULT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*router*": {"enable": False},  # Skip the MOE router
        "*output_layer*": {"enable": False},
        **_default_disabled_quantizer_cfg,
        "default": {"enable": False},
    },
    "algorithm": "max",
}

FP8_DYNAMIC_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": -1, "type": "dynamic", "enable": True},
        "*input_quantizer": {"num_bits": (4, 3), "axis": -1, "type": "dynamic", "enable": True},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*router*": {"enable": False},  # Skip the MOE router
        "*output_layer*": {"enable": False},
        **_default_disabled_quantizer_cfg,
        "default": {"enable": False},
    },
    "algorithm": "max",
}

FP8_WA_FP8_KV_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*router*": {"enable": False},  # Skip the MOE router
        "*bmm_quantizer": {"num_bits": (4, 3), "axis": None},
        "*output_layer*": {"enable": False},
        **_default_disabled_quantizer_cfg,
        "default": {"enable": False},
    },
    "algorithm": "max",
}

INT4_BLOCKWISE_WEIGHT_ONLY_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
        "*input_quantizer": {"enable": False},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*router*": {"enable": False},  # Skip the MOE router
        "*output_layer*": {"enable": False},
        **_default_disabled_quantizer_cfg,
        "default": {"enable": False},
    },
    "algorithm": "max",
}

NF4_REAL_QUANT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": 4,
            "block_sizes": {-1: 64, "scale_bits": 8, "scale_block_sizes": {-1: 256}},
            "enable": True,
            "fake_quant": False,
        },
        "*input_quantizer": {"enable": False},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*router*": {"enable": False},  # Skip the MOE router
        "*output_layer*": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": {"method": "real_quantize"},
}

INT4_AWQ_REAL_QUANT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": 4,
            "block_sizes": {-1: 128, "type": "static"},
            "enable": True,
            "fake_quant": False,
        },
        "*input_quantizer": {"enable": False},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*router*": {"enable": False},  # Skip the MOE router
        "*output_layer*": {"enable": False},
        **_default_disabled_quantizer_cfg,
        "default": {"enable": False},
    },
    "algorithm": {
        "method": "real_quantize",
        "additional_algorithm": {"method": "awq_lite", "alpha_step": 0.1},
    },
}

INT4_AWQ_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": 4,
            "block_sizes": {-1: 128, "type": "static"},
            "enable": True,
        },
        "*input_quantizer": {"enable": False},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*router*": {"enable": False},  # Skip the MOE router
        "*output_layer*": {"enable": False},
        **_default_disabled_quantizer_cfg,
        "default": {"enable": False},
    },
    "algorithm": {"method": "awq_lite", "alpha_step": 0.1},
    # "algorithm": {"method": "awq_full", "alpha_step": 0.1, "max_co_batch_size": 1024},
    # "algorithm": {"method": "awq_clip", "max_co_batch_size": 2048},
}

# W4A8 currently uses INT4 blockwise quantization (block size = 128) followed by FP8 quantization
# for weights. This could change in the future
W4A8_AWQ_BETA_CFG = {
    "quant_cfg": {
        "*weight_quantizer": [
            {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}, "enable": True},
            {"num_bits": (4, 3), "axis": None, "enable": True},
        ],
        "*input_quantizer": {"num_bits": (4, 3), "axis": -1, "enable": True},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*router*": {"enable": False},  # Skip the MOE router
        "*output_layer*": {"enable": False},
        **_default_disabled_quantizer_cfg,
        "default": {"enable": False},
    },
    "algorithm": "awq_lite",
}



choices: Set[str] = {
    "INT8_DEFAULT_CFG",
    "INT8_SMOOTHQUANT_CFG",
    "FP8_DEFAULT_CFG",
    "INT4_BLOCKWISE_WEIGHT_ONLY_CFG",
    "INT4_AWQ_CFG",
    "W4A8_AWQ_BETA_CFG",
    "NF4_REAL_QUANT_CFG",
    "INT4_AWQ_REAL_QUANT_CFG",
    "INT8_MIX",
}


class QuantizerAttributeConfig(ModeloptBaseConfig):
    """Quantizer attribute type."""

    enable: bool = ModeloptField(
        default=True,
        title="Enable quantizer.",
        description="""If True, enables the quantizer. If False, by-pass the quantizer and returns the input tensor.""",
    )

    num_bits: Union[int, Tuple[int, int]] = ModeloptField(
        default=8,
        title="An integer or a tuple of two integers specifying the number of quantization bits.",
        description="""`num_bits` can be:

        #. A positive integer argument for integer quantization. `num_bits` specify
            the number of bits used for integer quantization.

        #. Constant integer tuple (E,M) for floating point quantization emulating
            Nvidia's FPx quantization. E is the number of exponent bits and M is the number
            of mantissa bits. Supported FPx quantization formats: FP8 with (E=4, M=3).""",
    )

    @field_validator("num_bits")
    @classmethod
    def validate_num_bits(cls, v: Union[int, Tuple[int, int]]):
        """Validate `num_bits`."""
        if isinstance(v, tuple) and v not in [(4, 3)]:
            raise ValueError("Supported FPx quantization formats: FP8 (E=4, M=3).")
        return v

    axis: Optional[Union[int, Tuple[int, ...]]] = ModeloptField(
        default=None,
        title="None, integer or a tuple of integers specifying the axis to quantize.",
        description="""The specified axis/axes will have its own amax for
            computing scaling factor. If None (the default), use per tensor scale. Must be in the
            range [-rank(input_tensor), rank(input_tensor)). E.g. For a KCRS weight tensor,
            ``quant_axis=(0)`` will yield per channel scaling.""",
    )

    fake_quant: bool = ModeloptField(
        default=True,
        title="Enable fake quantization.",
        description="""If True, enable fake quantization.""",
    )

    unsigned: bool = ModeloptField(
        default=False,
        title="Enable unsigned quantization.",
        description="""If True, enable unsigned quantization. Used only for integer quantization.""",
    )

    narrow_range: bool = ModeloptField(
        default=False,
        title="Enable narrow range quantization.",
        description="""If True, enable narrow range quantization. Used only for integer quantization.""",
    )

    learn_amax: bool = ModeloptField(
        default=False,
        title="Enable learning amax.",
        description="""If True, enable learning amax.""",
    )

    type: str = ModeloptField(
        default="static",
        title="""Specify whether the quantization is static or dynamic.""",
        description="""The value is a string from ``["static", "dynamic"]``.
            If ``"dynamic"``, dynamic quantization will be enabled which does not collect any statistics during
            calibration.""",
        pattern=r"^static$|^dynamic$",
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v, info: ValidationInfo):
        """Validate type."""
        if v == "dynamic":
            assert info.data["learn_amax"] is False
        return v

    block_sizes: Optional[
        Dict[Union[int, str], Union[int, Tuple[int, int], str, Dict[int, int]]]
    ] = ModeloptField(
        default=None,
        title="Optional dictionary specifying block quantization parameters.",
        description="""The keys are the axes for block quantization and the
            values are block sizes for quantization along the respective axes. Keys must be in the
            range ``[-tensor.dim(), tensor.dim())``. Values, which are the block sizes
            for quantization must be positive integers.

            In addition, there can be special string keys ``"type"``, ``"scale_bits"`` and ``"scale_block_sizes"``.

            Key ``"type"`` should map to ``"dynamic"`` or ``"static"`` where ``"dynamic"``
            indicates dynamic block quantization and "static"
            indicates static calibrated block quantization. By default, the type is ``"static"``.

            Key ``"scale_bits"`` specify the quantization bits for the per-block quantization scale factor
            (i.e a double quantization scheme).

            Key ``"scale_block_sizes"`` specify the block size for double quantization.
            By default per-block quantization scale is not quantized.

            For example, ``block_sizes = {-1: 32}`` will quantize the last axis of the input tensor in
            blocks of size 32 with static calibration and ``block_sizes = {-1: 32, "type": "dynamic"}``
            will perform dynamic block quantization. If None, block
            quantization is not performed. ``axis`` must be None when ``block_sizes`` is not None.
        """,
    )

    @staticmethod
    def _get_block_quant_axes_and_sizes(block_sizes):
        if block_sizes is None:
            return None
        return {
            k: v
            for k, v in block_sizes.items()
            if k not in ["type", "scale_bits", "scale_block_sizes"]
        }

    @field_validator("block_sizes")
    @classmethod
    def validate_block_sizes(cls, v, info: ValidationInfo):
        """Validate block sizes."""
        if v is None:
            return v
        assert info.data["axis"] is None, "axis must be None when block_sizes is not None."
        if v.get("type", None) == "dynamic":
            assert (
                len(cls._get_block_quant_axes_and_sizes(v)) == 1
            ), "Dynamic block quantization only supports quantization last axis."
        for _k, _v in v.items():
            if isinstance(_k, str):
                assert _k in ["type", "scale_bits", "scale_block_sizes"]
            else:
                assert isinstance(_k, int) and isinstance(_v, int)
        return v

    trt_high_precision_dtype: str = ModeloptField(
        default="Float",
        title="TRT StronglyType requires all weights and amax to be in the same dtype.",
        description="""The value is a string from ``["Float", "Half", "BFloat16"]``.
            The QDQs will be assigned the appropriate data type, and this variable will only be
            used when the user is exporting the quantized ONNX model.""",
        pattern=r"^Float$|^Half$|^BFloat16$",
    )

    calibrator: Union[str, ConstructorLike] = ModeloptField(
        default="max",
        title="""Specify the calibrator to use.""",
        description="""The calibrator can be a string from ``["max", "histogram"]`` or a constructor
        to create a calibrator which subclasses :class:`_Calibrator <modelopt.torch.quantization.calib._Calibrator>`.
        See :meth:`standardize_constructor_args <modelopt.torch.utils.network.standardize_constructor_args>`
        for more information on how to specify the constructor.""",
    )

    @field_validator("calibrator")
    @classmethod
    def validate_calibrator(cls, v, info: ValidationInfo):
        """Validate calibrator."""
        if isinstance(v, str):
            assert v in ["max", "histogram"]
        return v


class QuantizeAlgorithmConfig(ModeloptBaseConfig):
    """Calibration algorithm config base."""

    method: str = ModeloptField(
        default="max",
        title="The calibration algorithm.",
        description="""The algorithm used for calibration. Supported algorithms include
        ``"max", "smoothquant", "awq_lite", "awq_full", and "awq_clip"``.""",
    )


class MaxCalibConfig(QuantizeAlgorithmConfig):
    """The config for max calibration algorithm.

    Max calibration estimates max values of activations or weights and use this max values
    to set the quantization scaling factor.
    See `Integer Quantization <https://arxiv.org/pdf/2004.09602>`_ for the concepts.
    """


class SmoothQuantCalibConfig(QuantizeAlgorithmConfig):
    """The config for ``smoothquant`` algorithm (SmoothQuant).

    SmoothQuant applies a smoothing factor which balances the scale of outliers in weights and activations.
    See `SmoothQuant paper <https://arxiv.org/pdf/2211.10438>`_ for more details.
    """

    alpha: Optional[float] = ModeloptField(
        default=1.0,
        ge=0.0,
        le=1.0,
        title="SmoothQuant hyper-parameter alpha.",
        description=(
            "This hyper-parameter controls the migration strength."
            "The migration strength is within [0, 1], "
            "a larger value migrates more quantization difficulty to weights."
        ),
    )


class AWQLiteCalibConfig(QuantizeAlgorithmConfig):
    """The config for ``awq_lite`` (AWQ lite) algorithm.

    AWQ lite applies a channel-wise scaling factor which minimizes the output difference after quantization.
    See `AWQ paper <https://arxiv.org/pdf/2306.00978>`_ for more details.
    """

    alpha_step: Optional[float] = ModeloptField(
        default=0.1,
        gt=0.0,
        le=1.0,
        title="Step size for the searching alpha.",
        description="The alpha will be searched from 0 to 1 with the step size specified.",
    )

    debug: Optional[bool] = ModeloptField(
        default=False,
        title="Debug mode.",
        description="If True, module's search metadata will be kept as a module attribute named `awq_lite`.",
    )


class AWQClipCalibConfig(QuantizeAlgorithmConfig):
    """The config for ``awq_clip`` (AWQ clip) algorithm.

    AWQ clip searches clipped amax for per-group quantization, This search requires much more compute
    compared to AWQ lite. To avoid any OOM, the linear layer weights are batched along the ``out_features``
    dimension of batch size ``max_co_batch_size``. AWQ clip calibration also takes longer than AWQ lite.
    """

    max_co_batch_size: Optional[int] = ModeloptField(
        default=1024,
        title="Maximum output channel batch size while searching clip values.",
        description="Reduce this number if CUDA Out of Memory error occurs.",
    )

    max_tokens_per_batch: Optional[int] = ModeloptField(
        default=64,
        title="Maximum tokens per batch while searching clip values.",
        description="""The total tokens used for clip search would be ``max_tokens_per_batch * number of batches``.
        Original AWQ uses a total of 512 tokens to search for clip values.""",
    )

    min_clip_ratio: Optional[float] = ModeloptField(
        default=0.5,
        gt=0.0,
        lt=1.0,
        title="Minimum clip ratio to search for.",
        description="""It should be in (0, 1.0). Clip will search for the optimal clipping value in the range
        ``[original block amax * min_clip_ratio, original block amax]``.""",
    )

    shrink_step: Optional[float] = ModeloptField(
        default=0.05,
        gt=0.0,
        le=1.0,
        title="Step size to search for clip values.",
        description="""It should be in range (0, 1.0]. The clip ratio will be searched from ``min_clip_ratio`` to 1
        with the step size specified.""",
    )

    debug: Optional[bool] = ModeloptField(
        default=False,
        title="Debug mode.",
        description="If True, module's search metadata will be kept as a module attribute named ``awq_clip``.",
    )


class AWQFullCalibConfig(AWQLiteCalibConfig, AWQClipCalibConfig):
    """The config for ``awq`` or ``awq_full`` algorithm (AWQ full).

    AWQ full performs ``awq_lite`` followed by ``awq_clip``.
    """

    debug: Optional[bool] = ModeloptField(
        default=False,
        title="Debug mode.",
        description=(
            "If True, module's search metadata will be kept as "
            "module attributes named ``awq_lite`` and ``awq_clip``."
        ),
    )


class RealQuantizeConfig(QuantizeAlgorithmConfig):
    """The config for real quantization config.

    The ``additional_algorithm`` will be used for calibration before quantizing weights into low precision.
    """

    additional_algorithm: Optional[
        Union[AWQLiteCalibConfig, AWQClipCalibConfig, AWQFullCalibConfig]
    ] = ModeloptField(
        default="",
        title="Additional algorithm for calibration before applying real quantization.",
        description="""The algorithm used for calibration. Supported algorithms include
        ``"awq_lite", "awq_full", and "awq_clip"``.""",
    )


QuantizeQuantCfgType = Dict[
    Union[str, Callable],
    Union[
        QuantizerAttributeConfig,
        List[QuantizerAttributeConfig],
        Dict[
            Union[str, Callable],
            Union[QuantizerAttributeConfig, List[QuantizerAttributeConfig]],
        ],
    ],
]

CalibAlgorithmCfgType = Union[
    MaxCalibConfig,
    SmoothQuantCalibConfig,
    AWQLiteCalibConfig,
    AWQClipCalibConfig,
    AWQFullCalibConfig,
    RealQuantizeConfig,
]


class QuantizeConfig(ModeloptBaseConfig):
    """Default configuration for ``quantize`` mode."""

    quant_cfg: QuantizeQuantCfgType = ModeloptField(
        default={"default": {"num_bits": 8, "axis": None}},
        title="Quantization configuration",
        validate_default=True,
    )

    algorithm: Union[str, CalibAlgorithmCfgType] = ModeloptField(
        default="max",
        title="Calibration algorithm",
        validate_default=True,
    )

    @field_validator("algorithm")
    @classmethod
    def validate_calibrator(cls, v, info: ValidationInfo):
        """Validate algorithm."""
        if isinstance(v, str):
            assert v in ["max", "smoothquant", "awq_lite", "awq_clip", "awq_full", "real_quantize"]
        return v


class _QuantizeExportConfig(ModeloptBaseConfig):
    """An empty config."""
