# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""TensorQuantizer Module."""
import contextlib
import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from packaging.version import Version
from torch import nn

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.qtensor import (
    BaseQuantizedTensor,
    INT4QTensor,
    NF4QTensor,
    QTensorWrapper,
)
from modelopt.torch.quantization.utils import is_torch_export_mode
from modelopt.torch.utils import standardize_constructor_args
from modelopt.torch.utils.distributed import DistributedProcessGroup

from ... import calib
from ... import utils as quant_utils
from ...tensor_quant import (
    fake_tensor_quant,
    scaled_e4m3,
)
from .clip import Clip

__all__ = ["TensorQuantizer", "SequentialQuantizer"]


class TensorQuantizer(nn.Module):
    """Tensor quantizer module.

    This module manages quantization and calibration of input tensor. It can perform fake (simulated quantization)
    or real quantization for various precisions and formats such as FP8 per-tensor, INT8 per-channel,
    INT4 per-block etc.

    If quantization is enabled, it calls the appropriate quantization functional and
    returns the quantized tensor. The quantized tensor data type will be same as the input tensor data type for
    fake quantization. During calibration mode, the module collects the statistics using its calibrator.

    The quantization parameters are as described in
    :class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>`. They can be set
    at initialization using ``quant_attribute_cfg`` or later by calling :meth:`set_from_attribute_config`.

    Args:
        quant_attribute_cfg: An instance of
            :class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>` or None.
            If None, default values are used.
        if_quant: A boolean. If True, quantization is enabled in the forward path.
        if_clip: A boolean. If True, clipping (with ``_learn_amax``) is enabled in the forward path.
        if_calib: A boolean. If True, calibration is enabled in the forward path.
        amax: None or an array like object such as list, tuple, numpy array, scalar
            which can be used to construct amax tensor.
    """

    def __init__(
        self,
        quant_attribute_cfg=None,
        if_quant=True,
        if_clip=False,
        if_calib=False,
        amax=None,
    ):
        """Initialize quantizer and set up required variables."""
        super(TensorQuantizer, self).__init__()
        quant_attribute_cfg = (
            quant_attribute_cfg if quant_attribute_cfg is not None else QuantizerAttributeConfig()
        )
        self.set_from_attribute_config(quant_attribute_cfg)

        self._scale = None if not quant_attribute_cfg.fake_quant else 1.0
        self._if_quant = if_quant
        self._if_clip = if_clip
        self._if_calib = if_calib
        self._enable_pre_quant_scale = True
        self._dequantize = False

        if amax is not None:
            self.amax = amax

        # Clip module consumes a lot of memory, so only create it if learn_amax is True
        if self._learn_amax:
            init_amax = self.amax if hasattr(self, "_amax") else 1.0
            self.clip = Clip(-init_amax, init_amax, learn_min=True, learn_max=True)
            self.enable_clip()

    def set_from_attribute_config(self, attribute_cfg: Union[QuantizerAttributeConfig, Dict]):
        """Set quantizer attributes from attribute_dict.

        The attributes are defined in
        :class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>`.
        """

        def _calibrator_setter(val):
            if val in ["max", "histogram"]:
                calib_cls = calib.MaxCalibrator if val == "max" else calib.HistogramCalibrator
                args, kwargs = (self._num_bits, self._axis, self._unsigned), {}
            else:
                calib_cls, args, kwargs = standardize_constructor_args(val)
            return calib_cls(*args, **kwargs)

        # Some attributes need custom handling.
        # By default, attributes from config are mapped to a name ``f"_{attribute}"``
        _custom_setters: Dict[str, Tuple[str, Callable]] = {
            "enable": ("_disabled", lambda val: val is False),
            "type": ("_dynamic", lambda val: val == "dynamic"),
            "calibrator": ("_calibrator", _calibrator_setter),
        }

        for attribute, val in attribute_cfg.items():
            assert (
                attribute in QuantizerAttributeConfig.model_fields
            ), f"{attribute} is not a valid `TensorQuantizer` attribute"
            _tq_attribute_name, _setter = _custom_setters.get(
                attribute, (f"_{attribute}", lambda v: v)
            )
            setattr(self, _tq_attribute_name, _setter(val))

    def dequantize(self, qtensor: BaseQuantizedTensor, dtype: torch.dtype):
        """De-quantize a real quantized tensor to a given dtype."""
        kwarg = {
            "scale": self._scale,
            "block_sizes": self.block_sizes,
            "double_scale": getattr(self, "_double_scale", None),
            "scale_zeros": getattr(self, "_scale_zeros", None),
        }
        return qtensor.dequantize(dtype, **kwarg)

    @property
    def num_bits(self):
        """Return num_bits for quantization."""
        return self._num_bits

    @num_bits.setter
    def num_bits(self, value):
        self._num_bits = value

    @property
    def maxbound(self):
        """Return maxbound for quantization."""
        if self._num_bits == (4, 3):
            return 448.0
        return (1 << (self._num_bits - 1 + int(self._unsigned))) - 1

    @property
    def unsigned(self):
        """Return True if unsigned quantization is used."""
        return self._unsigned

    @unsigned.setter
    def unsigned(self, value):
        self._unsigned = value

    @property
    def scale(self):
        """Return scale used for quantization."""
        return self._scale

    @property
    def pre_quant_scale(self):
        """Return pre_quant_scale used for smoothquant."""
        if not hasattr(self, "_pre_quant_scale") or not self._enable_pre_quant_scale:
            return None
        return self._pre_quant_scale

    @pre_quant_scale.setter
    def pre_quant_scale(self, value):
        assert value is not None, "pre_quant_scale cannot be set to None."
        assert (
            self._enable_pre_quant_scale
        ), "pre_quant_scale cannot be set when forward_with_pre_quant_scale is False."
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if not hasattr(self, "_pre_quant_scale"):
            self.register_buffer("_pre_quant_scale", value.clone().detach())
        else:
            if self._pre_quant_scale.shape != value.shape:
                raise RuntimeError("Changing shape when setting pre_quant_scale is not allowed.")
            self._pre_quant_scale.data.copy_(
                value.clone().detach().to(self._pre_quant_scale.device)
            )

    @property
    def amax(self):
        """Return amax for quantization."""
        if not hasattr(self, "_amax"):
            return None
        assert not self._dynamic, "Dynamic quantization does not have fixed amax"
        return self._amax

    @amax.setter
    def amax(self, value):
        assert value is not None, "amax cannot be set to None."

        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        if not hasattr(self, "_amax"):
            self.register_buffer("_amax", value.clone().detach())
        else:
            if self._amax.shape != value.shape:
                raise RuntimeError("Changing shape when setting amax is not allowed.")
            self._amax.data.copy_(value.clone().detach().to(self._amax.device))

    def reset_amax(self):
        """Reset amax to None."""
        if hasattr(self, "_amax"):
            delattr(self, "_amax")
        self._calibrator.reset()

    @property
    def step_size(self):
        """Return step size for integer quantization."""
        if not hasattr(self, "_amax"):
            warnings.warn("step_size is undefined under dynamic amax mode!")
            return None
        assert isinstance(
            self._num_bits, int
        ), "Step size is not defined for non-integer quantization."
        return self._amax / (2.0 ** (self._num_bits - 1 + int(self._unsigned)) - 1.0)

    @property
    def axis(self):
        """Return axis for quantization."""
        return self._axis

    @axis.setter
    def axis(self, value):
        self._axis = value

    @property
    def block_sizes(self):
        """Return block_sizes for quantization."""
        return self._block_sizes

    @block_sizes.setter
    def block_sizes(self, value):
        self._axis = None
        self._block_sizes = value

    @property
    def fake_quant(self):
        """Return True if fake quantization is used."""
        return self._fake_quant

    @property
    def narrow_range(self):
        """Return True if symmetric integer range for signed quantization is used."""
        return self._narrow_range

    @narrow_range.setter
    def narrow_range(self, value):
        self._narrow_range = value

    @property
    def is_enabled(self):
        """Return true if the modules is not disabled."""
        return not self._disabled

    def disable(self):
        """Bypass the module.

        Neither of calibration, clipping and quantization will be performed if the module is disabled.
        """
        self._disabled = True

    def enable(self):
        """Enable the module."""
        self._disabled = False

    @property
    def trt_high_precision_dtype(self):
        """Return True if FP16 AMAX is used when exporting the model."""
        return self._trt_high_precision_dtype

    @trt_high_precision_dtype.setter
    def trt_high_precision_dtype(self, value):
        self._trt_high_precision_dtype = value

    def disable_clip(self):
        """Disable clip stage."""
        self._if_clip = False
        self.clip.clip_value_min.requires_grad = False
        self.clip.clip_value_max.requires_grad = False

    def enable_clip(self):
        """Enable clip stage."""
        if not self._learn_amax:
            raise ValueError("learn_amax is False. Cannot enable clip.")
        self.clip.clip_value_min.requires_grad = True
        self.clip.clip_value_max.requires_grad = True
        self._if_clip = True

    def disable_calib(self):
        """Disable calibration."""
        self._if_calib = False

    def enable_calib(self):
        """Enable calibration."""
        if self._calibrator is None:
            raise ValueError("Calibrator was not created, cannot enable calibration.")

        # Dynamic quantization does not need calibration.
        if self._dynamic:
            return
        self._if_calib = True

    def disable_quant(self):
        """Disable quantization."""
        self._if_quant = False

    def enable_quant(self):
        """Enable quantization."""
        self._if_quant = True

    def load_calib_amax(self, *args, **kwargs):
        """Load amax from calibrator.

        Updates the amax buffer with value computed by the calibrator, creating it if necessary.
        ``*args`` and ``**kwargs`` are directly passed to ``compute_amax``, except ``"strict"`` in
        ``kwargs``. Refer to ``compute_amax`` for more details.
        """
        assert not self._dynamic, "Dynamic quantization does not need calibration."

        strict = kwargs.pop("strict", True)
        if getattr(self, "_calibrator", None) is None:
            raise RuntimeError("Calibrator not created.")
        calib_amax = self._calibrator.compute_amax(*args, **kwargs)
        if calib_amax is None:
            err_msg = (
                "Calibrator returned None. This usually happens when calibrator hasn't seen any"
                " tensor."
            )
            if not strict:
                warnings.warn(err_msg)
                warnings.warn("Set amax to NaN!")
                calib_amax = torch.tensor(math.nan)
            else:
                raise RuntimeError(
                    err_msg
                    + " Passing 'strict=False' to `load_calib_amax()` will ignore the error."
                )
        if not hasattr(self, "_amax"):
            self.register_buffer("_amax", calib_amax.clone().detach())
        else:
            self._amax.data.copy_(calib_amax.clone().detach())

    def init_learn_amax(self):
        """Initialize learned amax from fixed amax."""
        if self._learn_amax is False:
            raise RuntimeError("Called init_learn_amax with learn_amax=False.")

        if self._amax.numel() != 1:
            warnings.warn("Per channel learned amax not supported. Initializing with max(amax).")
            init_amax = torch.max(self._amax)
        else:
            init_amax = self._amax
        self.clip.clip_value_min.data.copy_(-init_amax.clone().detach())
        self.clip.clip_value_max.data.copy_(init_amax.clone().detach())

    def _get_amax(self, inputs):
        """Get amax from buffer or compute it dynamically."""
        if hasattr(self, "_amax"):
            amax = self._amax
        else:
            if self._axis is None:
                reduce_axis = None
            else:
                reduce_axis = []
                # Swap axis to reduce
                axis = self._axis if isinstance(self._axis, (list, tuple)) else [self._axis]
                for i in range(inputs.dim()):
                    if not (i in axis or (i - inputs.dim()) in axis):
                        reduce_axis.append(i)
            amax = quant_utils.reduce_amax(inputs, axis=reduce_axis, keepdims=True).detach()

        amax = amax.data
        return amax

    def _validate_amax(self, amax):
        # Dynamic control flow is not supported by torch dynamo
        if not is_torch_export_mode():
            assert torch.all(amax >= 0) and not torch.any(
                torch.isinf(amax)
            ), f"Got invalid amax: {amax}"

    def _real_quantize(self, inputs):
        assert (
            self._num_bits == 4 and self._block_sizes
        ), "Only support 4-bit block-wise real quantization."
        if self._block_sizes.get("scale_bits", 0) == 8 and self._block_sizes.get(
            "scale_block_sizes", None
        ):
            # NF4 double quantization
            # Return real quantized tensor class and store scales inside the TensorQuantizer
            outputs, scales = NF4QTensor.quantize(inputs, self._block_sizes[-1])
            self._scale, self._double_scale, self._scale_zeros = NF4QTensor.double_quantization(
                scales,
                self._block_sizes["scale_block_sizes"][-1],
                self._block_sizes["scale_bits"],
            )
        else:
            outputs, self._scale = INT4QTensor.quantize(inputs, self._block_sizes[-1])

        return outputs

    def _quant_forward(self, inputs):
        """Quantized forward pass."""
        amax = None
        if not self._dequantize:
            if self._learn_amax:
                inputs = self.clip(inputs)
                amax = torch.max(-self.clip.clip_value_min, self.clip.clip_value_max).detach()
            else:
                amax = self._get_amax(inputs)

            self._validate_amax(amax)

        if self._fake_quant:
            if isinstance(self._num_bits, tuple):
                E, M = self._num_bits  # noqa: N806
                outputs = scaled_e4m3(
                    inputs, self._get_amax(inputs), E, M, self._trt_high_precision_dtype
                )
            else:
                outputs = fake_tensor_quant(
                    inputs,
                    amax,
                    self._num_bits,
                    self._unsigned,
                    self._narrow_range,
                    self._trt_high_precision_dtype,
                )
        else:
            # Real quantize
            if not self._dequantize:
                outputs = self._real_quantize(inputs)
                self._dequantize = True
            else:
                # De-quantize
                if isinstance(inputs, QTensorWrapper):
                    inputs = inputs._qtensor
                assert isinstance(
                    inputs, BaseQuantizedTensor
                ), "Expected input as real quantized tensors."
                return self.dequantize(inputs, inputs._original_meta_tensor.dtype)
        return outputs

    def _check_onnx_readiness(self, inputs):
        """Check if quantizer is ready for ONNX export."""
        assert hasattr(self, "_amax"), (
            "Quantizer has not been calibrated. ONNX export requires the quantizer to be"
            " calibrated.Calibrate and load amax before exporting to ONNX."
        )

        if self._if_calib:
            warnings.warn(
                "Quantizer is in calibration mode. "
                "Please complete calibration before exporting to ONNX for correct results."
            )

        amax = self._get_amax(inputs)

        # We only support scalar amax for E4M3 ONNX export
        if isinstance(self.num_bits, tuple):
            assert amax.numel() == 1, (
                "E4M3 supports ONNX export only for per-tensor quantization."
                " Per-tensor quantization requires scalar amax. "
                f"Received non-scalar amax of shape: {amax.shape}"
            )

        if self.block_sizes is not None:
            raise Exception("Blockquant does not support ONNX export.")

    def _setup_for_blockquant(self, inputs: torch.Tensor):
        # Get reshape sizes and paddings for block-quantization
        def get_axis_quant_params(ax):
            ax = ax if ax in self.block_sizes else ax - inputs.dim()
            bsize = self.block_sizes.get(ax, None)
            padding, ax_slice = None, None
            if bsize is not None and inputs.shape[ax] % bsize != 0:
                padding = (bsize - (inputs.shape[ax] % bsize), 0)
                ax_slice = slice(inputs.shape[ax])
            return bsize, padding, ax_slice

        def set_quant_params(axis, block_reshape_size, padding, slices, amax_shape=None):
            self._axis = tuple(axis)
            if hasattr(self, "_calibrator"):
                self._calibrator._axis = self._axis
            self._original_shape = inputs.shape
            self._block_reshape_size = torch.Size(block_reshape_size)
            if padding is not None:
                self._padding = tuple(padding)
                self._original_shape = F.pad(inputs, self._padding, "constant", 0).shape
            if slices is not None:
                self._slices = slices
            if amax_shape:
                self._amax_shape_for_export = amax_shape

        # Reshape size have already been set
        if hasattr(self, "_block_reshape_size"):
            return

        reshape_size, quantize_axis, paddings, slices = [], [], [], []

        # special handling for block-quantization along the last axis:
        # flatten the input for faster execution
        if (self.block_sizes.get(inputs.dim() - 1, None) or self.block_sizes.get(-1, None)) and len(
            QuantizerAttributeConfig._get_block_quant_axes_and_sizes(self.block_sizes)
        ) == 1:
            bsize, padding, ax_slice = get_axis_quant_params(inputs.dim() - 1)
            slices = None if ax_slice is None else (*(slice(None),) * (inputs.dim() - 1), ax_slice)
            padding = padding if not padding else tuple(reversed(padding))
            amax_shape_for_export = (*(inputs.shape[:-1]), -1)
            set_quant_params((0,), (-1, bsize), padding, slices, amax_shape_for_export)
            return

        for ax in range(inputs.dim()):
            bsize, padding, ax_slice = get_axis_quant_params(ax)
            paddings.append(padding)
            slices.append(ax_slice)
            if bsize is not None:
                reshape_size.extend([math.ceil(inputs.shape[ax] / bsize), bsize])
                quantize_axis.extend([True, False])
            else:
                reshape_size.append(inputs.shape[ax])
                quantize_axis.append(True)

        quant_axis = [i for i in range(len(quantize_axis)) if quantize_axis[i]]

        if all(s is None for s in slices):
            slices = None
        else:
            slices = [s if s else slice(None) for s in slices]

        if all(p is None for p in paddings):
            paddings = None
        else:
            new_paddings = []
            for padding in paddings:
                if not (new_paddings or padding):
                    continue
                new_paddings.extend(padding if padding else (0, 0))
            paddings = tuple(reversed(new_paddings))

        set_quant_params(quant_axis, reshape_size, paddings, slices)

    def _process_for_blockquant(self, inputs: torch.Tensor):
        if hasattr(self, "_padding"):
            inputs = F.pad(inputs, self._padding, "constant", 0)
        assert inputs.shape == self._original_shape, (
            f"Input shape has changed from {self._original_shape} to {inputs.shape}."
            " Block-quantization requires a fixed input shape."
        )
        inputs = inputs.reshape(self._block_reshape_size)
        return inputs

    def _reset_to_original_shape(self, outputs: torch.Tensor):
        outputs = outputs.reshape(self._original_shape)
        if hasattr(self, "_slices"):
            outputs = outputs[self._slices]
        return outputs

    def export_amax(self) -> Optional[torch.Tensor]:
        """Export correctly formatted/shaped amax."""
        if self.block_sizes is not None and self.block_sizes.get("type", None) == "dynamic":
            raise NotImplementedError(
                "Exporting amax for dynamic block quantization is not supported."
            )

        if self.amax is None:
            return None

        if not hasattr(self, "_amax_shape_for_export"):
            amax = self.amax
        else:
            amax = self.amax.reshape(self._amax_shape_for_export)
        amax[amax == 0] = self.maxbound
        clamp_min, clamp_max = torch.finfo(amax.dtype).tiny, torch.finfo(amax.dtype).max
        amax = amax.clamp(min=clamp_min, max=clamp_max)

        self._validate_amax(amax)

        if self.block_sizes is None:
            # tensorrt_llm assumes the scaling_factor dim >= 1 for per-tensor.
            if self.axis is None:
                amax = amax.unsqueeze(0)

            # If single-axis quantization, squeeze amax
            elif isinstance(self.axis, int) or (
                isinstance(self.axis, (list, tuple)) and len(self.axis) == 1
            ):
                amax = amax.squeeze()

        return amax

    def forward(self, inputs):
        """Apply tensor_quant function to inputs.

        Args:
            inputs: A Tensor of type float32/float16/bfloat16.

        Returns:
            outputs: A Tensor of type output_dtype
        """
        if isinstance(inputs, BaseQuantizedTensor):
            assert self._dequantize, "No dequantization stats in the tensor quantizer."
            return self._quant_forward(inputs)

        # Early return if nothing is collected during the forward (e.g. MoE)
        if len(inputs) == 0:
            return inputs

        # Activation scaling for smoothquant
        if self.pre_quant_scale is not None:
            inputs = inputs * self.pre_quant_scale

        if self._disabled:
            return inputs

        # GLOBALS could break TorchDynamo for some Pytorch versions (i.e., 2.3.0)
        if not is_torch_export_mode():
            # GLOBALS.in_onnx_export is only available in torch >= 1.13
            if Version(torch.__version__) >= Version("1.13"):
                from torch.onnx._globals import GLOBALS

                if GLOBALS.in_onnx_export:
                    self._check_onnx_readiness(inputs)

        if (
            self.block_sizes is not None
            and not self.block_sizes.get("type", None) == "dynamic"
            and self._fake_quant
        ):
            # Dynamic block quantization is handled seperately by the quantization kernels
            self._setup_for_blockquant(inputs)
            inputs = self._process_for_blockquant(inputs)

        outputs = inputs

        if self._if_calib and not self._dynamic:
            if self._calibrator is None:
                raise RuntimeError("Calibrator was not created.")
            # Shape is only known when it sees the first tensor
            self._calibrator.collect(inputs)

        if self._if_clip:
            if not self._learn_amax:
                raise RuntimeError("Clip without learning amax is not implemented.")
            outputs = self.clip(inputs)

        if self._if_quant:
            outputs = self._quant_forward(inputs)

        if (
            self.block_sizes is not None
            and not self.block_sizes.get("type", None) == "dynamic"
            and self._fake_quant
        ):
            outputs = self._reset_to_original_shape(outputs)

        return outputs

    def _short_amax(self, fmt=".4f"):
        """Short description of amax.

        Returns:
            'dynamic': if _amax is not registered
            'amax': if _amax is per-tensor
            '[min, max](size)': if _amax is per-channel
        """
        if not hasattr(self, "_amax"):
            return "dynamic"
        if self._amax is None:
            return "None"
        if self._amax.numel() == 1:
            return f"{self._amax.item():{fmt}}"
        return (
            f"[{self._amax.min().item():{fmt}},"
            f" {self._amax.max().item():{fmt}}]({self._amax.numel()})"
        )

    def extra_repr(self):
        """Set the extra information about this module."""
        if self._disabled:
            return "disabled"
        s = f"{'unsigned ' if self._unsigned else ''}{self._num_bits} bit"
        s += " narrow" if (self._narrow_range) else ""
        s += " fake" if (self._fake_quant) else ""
        if self.block_sizes is not None:
            s += f" block_sizes={self._block_sizes},"
        else:
            s += f" axis={self._axis}" if self._axis is not None else " per-tensor"
        s += f" amax={self._short_amax()}"
        s += " pre_quant_scale" if self.pre_quant_scale is not None else ""
        s += " learned" if (self._learn_amax) else ""
        s += (
            f" calibrator={self._calibrator.__class__.__name__}"
            if (self._calibrator is not None)
            else ""
        )
        s += " quant" if (self._if_quant) else ""
        s += " clip" if (self._if_clip) else ""
        s += " calib" if (self._if_calib) else ""
        return s

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Overloaded module function.

        Adds warnings during state_dict loading.
        A workaround is implemented for loading amax from checkpoint and only supports CUDA.

        Args:
            state_dict: A dict containing the state of the top level module
            prefix: A string that prefixes all of this modules state in state_dict, e.g. 'model.conv1.'
        """
        dst_has_amax = "_amax" in self._buffers
        src_has_amax = prefix + "_amax" in state_dict

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not src_has_amax and dst_has_amax:
            warnings.warn(f"{prefix[:-1]}: No amax in state_dict.")
        elif src_has_amax and not dst_has_amax:
            warnings.warn(
                f"{prefix[:-1]}: No '_amax' buffer to load amax into."
                " '_amax` will be created as WAR for now. "
                "This behavior will change in future."
            )
            self.register_buffer("_amax", state_dict[prefix + "_amax"].clone().detach().to(device))
        elif (
            src_has_amax
            and dst_has_amax
            and (self._amax.shape != state_dict[prefix + "_amax"].shape)
        ):
            # This is a workaround to support the sharded checkpoint loading in Megatron.
            # The sharded checkpoint has the amax in a different shape.
            assert self._amax.numel() == state_dict[prefix + "_amax"].numel(), (
                f" {prefix[:-1]}: amax state cannot be loaded. Expected {self._amax.shape}, got"
                f" {state_dict[prefix + '_amax'].shape}"
            )
            state_dict[prefix + "_amax"] = state_dict[prefix + "_amax"].view(self._amax.shape)

        dst_has_pre_quant_scale = "_pre_quant_scale" in self._buffers
        src_has_pre_quant_scale = prefix + "_pre_quant_scale" in state_dict

        if not src_has_pre_quant_scale and dst_has_pre_quant_scale:
            warnings.warn(f"{prefix[:-1]}: No pre_quant_scale in state_dict.")
        elif src_has_pre_quant_scale and not dst_has_pre_quant_scale:
            warnings.warn(
                f"{prefix[:-1]}: No '_pre_quant_scale' buffer to load pre_quant_scale into."
                " '_pre_quant_scale` will be created as WAR for now. "
                "This behavior will change in future."
            )
            self.register_buffer(
                "_pre_quant_scale",
                state_dict[prefix + "_pre_quant_scale"].clone().detach().to(device),
            )

        super(TensorQuantizer, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _get_skip_properties_for_modelopt_state(self):
        return {"clip", "_calibrator", "_original_shape", "_block_reshape_size", "_padding"}

    def _get_properties_for_modelopt_state(self):
        return (
            self.__dict__.keys()
            - nn.Module().__dict__.keys()
            - self._get_skip_properties_for_modelopt_state()
        )

    def get_modelopt_state(self) -> Dict[str, Any]:
        """Get meta state to be saved in checkpoint."""
        modelopt_state = {}
        for k in self._get_properties_for_modelopt_state():
            modelopt_state[k] = getattr(self, k)

        if hasattr(self, "_amax"):
            modelopt_state["_has_amax"] = True

        if hasattr(self, "_pre_quant_scale"):
            modelopt_state["_has_pre_quant_scale"] = True

        if hasattr(self, "clip"):
            modelopt_state["_init_clip"] = True

        return modelopt_state

    def set_from_modelopt_state(self, modelopt_state, prefix=""):
        """Set meta state from checkpoint."""
        # Set all properties except the skip properties; this is done for backward compatibility
        for key in modelopt_state.keys() - self._get_skip_properties_for_modelopt_state():
            setattr(self, key, modelopt_state[key])

        # Set the calibrator properties
        # TODO: This might not be sufficient for the custom calibrators - however there is no use-case for it yet
        for key in ["_num_bits", "_axis", "_unsigned"]:
            setattr(self._calibrator, key, getattr(self, key))

        if "_init_clip" in modelopt_state:
            # clip min and max parameters will be loaded from checkpoint
            self.clip = Clip(-1.0, 1.0, learn_min=True, learn_max=True)

        # Create a temporary variable to indicate if the quantizer had amax in the checkpoint
        self._has_amax = modelopt_state.get("_has_amax", "_amax" in modelopt_state)

        # Create a temporary variable to indicate if the quantizer had pre_quant_scale in the checkpoint
        self._has_pre_quant_scale = modelopt_state.get(
            "_has_pre_quant_scale", "_pre_quant_scale" in modelopt_state
        )

    def clean_up_after_set_from_modelopt_state(self, prefix=""):
        """Clean up temporary variables created during set_from_modelopt_state."""
        warning_msg = (
            f"Could not initialize the quantizer states for {prefix}. The quantizer"
            " states after `load_state_dict` could be in the wrong device. Please move"
            " the modules to the correct device after loading the state dict."
        )

        if hasattr(self, "_has_amax"):
            if self._has_amax and self.amax is None:
                warnings.warn(warning_msg, UserWarning)
            delattr(self, "_has_amax")

        if hasattr(self, "_has_pre_quant_scale"):
            if self._has_pre_quant_scale and self.pre_quant_scale is None:
                warnings.warn(warning_msg, UserWarning)
            delattr(self, "_has_pre_quant_scale")

    def sync_amax_across_distributed_group(self, parallel_group: DistributedProcessGroup):
        """Synchronize the amax across all ranks in the given group."""
        if parallel_group.is_initialized() and self.amax is not None:
            try:
                dist.all_reduce(self.amax, op=dist.ReduceOp.MAX, group=parallel_group.group)
            except RuntimeError as e:
                # This error happens if the distributed backend is using GPU and
                # the tensor is not on GPU (or vice versa).
                warnings.warn(
                    (
                        f"Failed to synchronize amax: {e}, probably because the tensor is on a device which is not"
                        "supported by the current distributed backend. This warning can be ignored"
                        "if happening during modelopt restore."
                    )
                )


class SequentialQuantizer(nn.Sequential):
    """A sequential container for  :class:`TensorQuantizer` modules.

    This modules is used to quantize a tensor in multiple formats sequentially. It takes as input
    :class:`TensorQuantizer` modules and containerize them similar to :class:`torch.nn.Sequential`.

    Args:
        quantizers (TensorQuantizer): :class:`TensorQuantizer` modules to be added to the container.

    """

    def __init__(self, *quantizers: TensorQuantizer):  # noqa: N803
        """Initialize SequentialQuantizer module."""
        assert not any(
            not isinstance(q, TensorQuantizer) for q in quantizers
        ), "All quantizers must be a TensorQuantizer."
        super().__init__(*quantizers)

    def get_modelopt_state(self) -> Dict[str, Any]:
        """Get meta state to be saved in checkpoint."""
        return {"num_quantizers": len(self), "is_sequential_quantizer": True}

    def disable(self):
        """Disable the quantizer modules."""
        for quantizer in self:
            quantizer.disable()

    def set_from_attribute_config(self, attributes: List[Dict[str, Any]]):
        """Set the attributes of contained quantizers from a list of attribute_dicts."""
        for attribute, quantizer in zip(attributes, self):
            quantizer.set_from_attribute_config(attribute)

    @staticmethod
    @contextlib.contextmanager
    def replace_sequential_quantizer_with_single_quantizer(model, indx: int = 0):
        """Replace instances of :class:`SequentialQuantizer` in the model with single quantizers.

        The quantizer indexed by ``indx`` from the sequential quantizer is used to replace it.
        This method is useful for individually calibrating the quantizers in a sequential quantizer.
        """
        for name, module in list(model.named_modules()):
            if isinstance(module, SequentialQuantizer):
                assert len(module) > indx
                parent_module = model.get_submodule(name.rpartition(".")[0])
                setattr(parent_module, "_original_" + name.rpartition(".")[-1], module)
                setattr(parent_module, name.rpartition(".")[-1], module[indx])

        yield

        for name, module in list(model.named_modules()):
            if isinstance(module, SequentialQuantizer) and "_original_" in name.rpartition(".")[-1]:
                parent_module = model.get_submodule(name.rpartition(".")[0])
                original_name = name.rpartition(".")[-1].replace("_original_", "")
                setattr(parent_module, original_name, module)
                delattr(parent_module, name.rpartition(".")[-1])

    @staticmethod
    def tensor_quantizer_iterator(quantizers):
        """Iterator for the quantizers in the container (but yield itself if its a TensorQuantizer)."""
        if quantizers is None:
            return
        if isinstance(quantizers, TensorQuantizer):
            yield quantizers
        elif isinstance(quantizers, SequentialQuantizer):
            for quantizer in quantizers:
                yield quantizer
        else:
            raise ValueError("Invalid quantizer type.")
