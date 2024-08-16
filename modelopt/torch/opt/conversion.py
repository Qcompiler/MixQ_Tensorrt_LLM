# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Module to handle model converting and restoring for optimization methods.

When applying a model optimization algorithm, we usually need to modify the model in each step
(mode) of the algorithm. This module provides the state manager, which is a standardized interface
(class) to record and store state information in the model.

Op top of the state manager, this module provides utilities to save a history of these modifications
("modelopt state dict") and restoring a unmodified model to the state indicated in the state dict.
"""

import copy
import os
import warnings
from collections import deque
from typing import Any, BinaryIO, Deque, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from modelopt import __version__
from modelopt.torch.utils import ModelLike, init_model_from_model_like, unwrap_model

from .config import ConfigDict, ModeloptBaseConfig
from .mode import (
    MetadataDict,
    ModeLike,
    ModeState,
    ModeType,
    _ModeDescriptor,
    _ModeRegistryCls,
    get_mode_config,
)

__all__ = [
    "ModeloptStateManager",
    "apply_mode",
    "modelopt_state",
    "save",
    "restore_from_modelopt_state",
    "restore",
]

ModeloptStateList = List[Tuple[str, ModeState]]  # state data structure for multiple modes


class ModeloptStateManager:
    """A class to handle the modelopt state stored for each mode correspondig to a task/mode."""

    _state_key = "_modelopt_state"

    def __init__(self, model: Optional[nn.Module] = None, init_state: bool = False) -> None:
        """Initialize state manager.

        Args:
            model: Module that has modelopt_state stored. If None, a fake module is created to store
                any state that might be added with the manager.
            init_state: Whether to initialize the modelopt state for the model if it does not exist.
        """
        # just assume fake module for easy implementation below
        if not model:
            model = nn.Module()
            init_state = True  # always initialize fake module

        # initialize modelopt state if desired. Note that by default we don't do that to avoid
        # accidentally modifying a user-provided model.
        if init_state:
            assert not hasattr(model, self._state_key), "Model already has modelopt state!"
            setattr(model, self._state_key, [])

        # sanity check that root module has modelopt state now
        assert self.is_converted(model, is_root=True), "Model must have modelopt state!"

        # store reference to state
        self._state: ModeloptStateList = getattr(model, self._state_key)

    @property
    def has_state(self) -> bool:
        """Return whether the model has a non-trivial modelopt state."""
        return bool(self._state)

    @classmethod
    def is_converted(cls, model: nn.Module, is_root: bool = False) -> bool:
        """Check if model is converted.

        Args:
            model: A model to be checked for state/metadata from the convert process.
            is_root: Additionally check whether the module with state is the root module.

        Returns:
            True if the model contains modelopt state indicating that it has been converted.

        This method raises an assertion when multiple modelopt_states are detected or when is_root is
        set to True but the module with state is not the root module.
        """
        # check for submodules with state
        mods_with_state = [name for name, m in model.named_modules() if hasattr(m, cls._state_key)]
        # check if there is multiple submodules with state
        assert len(mods_with_state) <= 1, "Model has multiple modelopt states!"
        is_converted = bool(mods_with_state)

        # check if mod with state is root module if desired
        if is_converted:
            assert (
                not is_root or mods_with_state[0] == ""
            ), "Model has modelopt state but not the root!"

        return is_converted

    # TODO: consider renaming state_dict???
    def state_dict(self) -> ModeloptStateList:
        """Return the metadata of the model."""
        return self._state

    def load_state_dict(self, state_dict: ModeloptStateList) -> None:
        """Load the provided ``state_dict`` to the modelopt_state."""
        assert not self.has_state, "Cannot load state_dict if there is already one."

        # make sure we operate on deepcopy
        state_dict = copy.deepcopy(state_dict)
        # add modes one-by-one
        for m_str, m_state in state_dict:
            # adds config and metadata with sanity checks
            config = self.get_config_class(m_str, m_state["config"])
            self.add_mode(m_str, config, m_state["metadata"])

        # overwrite state manually afterwards to ensure exact consistency with provided state_dict
        self._state.clear()
        self._state.extend(state_dict)

    @classmethod
    def transfer_state_dict(cls, model_from: nn.Module, model_to: nn.Module) -> None:
        """Transfer the state (same instance) from one model to another."""
        manager_from = ModeloptStateManager(model_from, init_state=False)  # state must exist
        manager_to = ModeloptStateManager(model_to, init_state=True)  # state must NOT exist

        # transfer state_dict (this uses sanity checks + deepcopy)
        manager_to.load_state_dict(manager_from.state_dict())

        # manually set the state dict to be the exact same instance
        setattr(model_to, cls._state_key, manager_from.state_dict())
        manager_to = ModeloptStateManager(model_to, init_state=False)  # state must exist now

        # remove state from model_from
        delattr(model_from, cls._state_key)

    def modes_with_states(
        self,
    ) -> Iterator[Tuple[_ModeDescriptor, ModeloptBaseConfig, MetadataDict]]:
        """Yield the mode together with the full config and metadata from the state."""
        for m_str, m_state in self._state:
            config = self.get_config_class(m_str, m_state["config"])
            yield _ModeRegistryCls.get_from_any(m_str), config, m_state["metadata"]

    @property
    def last_mode(self) -> Optional[_ModeDescriptor]:
        """Return the last mode applied to the model (last stored mode)."""
        return _ModeRegistryCls.get_from_any(self._state[-1][0]) if self._state else None

    @property
    def _last_metadata(self) -> MetadataDict:
        """Return the metadata of the last mode applied to the model (must exist!)."""
        return self._state[-1][1]["metadata"]

    @property
    def _last_config(self) -> ModeloptBaseConfig:
        """Return the config of the last mode applied to the model (must exist!)."""
        return self.get_config_class(self._state[-1][0], self._state[-1][1]["config"])

    @_last_config.setter
    def _last_config(self, config: ModeloptBaseConfig) -> None:
        """Set the config of the last mode applied to the model (must exist!)."""
        self._state[-1][1]["config"] = config.model_dump()

    @property
    def _export_stack(self) -> Deque[Tuple[str, str]]:
        """Infer the stack of export modes that still must be applied from existing modes.

        Returns:
            A deque of tuples of the form ``(mode_str, export_mode_str)`` representing the mode
            which requires an export mode and the export mode itself.
        """
        stack = deque()
        for m, _, _ in self.modes_with_states():
            if m.export_mode:
                stack.append((str(m), m.export_mode))
            elif m.is_export_mode:
                assert str(m) == stack.pop()[1], "Inconsistent export stack!"
        return stack

    @staticmethod
    def get_config_class(mode: ModeType, config: ConfigDict) -> ModeloptBaseConfig:
        """Standardize the provided config to the corresponding config class."""
        # validate and standardize to the config class and return
        return _ModeRegistryCls.get_from_any(mode).config_class(**config)

    def check_mode(self, mode: ModeType) -> None:
        """Check if the proposed mode is compatible with the current state."""
        # standardize mode to descriptor
        mode_d = _ModeRegistryCls.get_from_any(mode)

        # check for export mode compatibility
        export_stack = self._export_stack
        if mode_d.is_export_mode:
            assert (
                export_stack and str(mode_d) == export_stack[-1][1]
            ), f"Cannot add {mode_d} according to the current export stack: {export_stack}."

        # sanity checks for next mode incompatibilities according to the current last mode
        last_mode = self.last_mode
        if last_mode:
            assert last_mode.next_modes is None or str(mode_d) in last_mode.next_modes, (
                f"Cannot add {mode_d} after {last_mode}! Next modes of {last_mode} are"
                f" {last_mode.next_modes}."
            )

        # sanity checks for next mode incompatible with last mode in the stack. These
        # incompatibilities still apply since we did not apply corresponding export mode yet.
        if export_stack:
            # last_m_stack := last mode on the stack that has a corresponding export mode
            last_m_stack = _ModeRegistryCls.get_from_any(export_stack[-1][0])
            assert last_m_stack.next_modes is None or str(mode_d) in last_m_stack.next_modes, (
                f"Cannot add {mode_d} after {last_m_stack}! Next modes of {last_m_stack} are"
                f" {last_m_stack.next_modes}."
            )

    def add_mode(self, mode: ModeType, config: ModeloptBaseConfig, metadata: MetadataDict) -> None:
        """Add mode and update state in-place.

        Note that self._state is a list (preserves insertion order of keys) and we can therefore
        recall the order of modes!
        """
        # standardize mode to descriptor
        mode_d = _ModeRegistryCls.get_from_any(mode)

        # sanity checks for mode incompatibilities
        self.check_mode(mode_d)

        # store mode information
        m_state: ModeState = {"config": config.model_dump(), "metadata": metadata}

        self._state.append((str(mode_d), m_state))

    def update_last_state_before_new_mode(self, model: nn.Module) -> None:
        """Update the metadata and config of the last mode applied to the model."""
        last_mode = self.last_mode
        if last_mode is not None:
            last_config = self._last_config
            last_mode.update_for_new_mode(model, last_config, self._last_metadata)
            self._last_config = last_config

    def update_last_state_before_save(self, model: nn.Module) -> None:
        """Update the metadata and config of the last mode applied to the model."""
        last_mode = self.last_mode
        if last_mode is not None:
            last_config = self._last_config
            last_mode.update_for_save(model, last_config, self._last_metadata)
            self._last_config = last_config


class ApplyModeError(RuntimeError):
    """Error raised when applying a mode to a model fails."""


class ModelLikeModule(nn.Module):
    """Just a temp module type to store the initialization recipe for the actual model."""

    def __init__(self, modellike: ModelLike) -> None:
        super().__init__()
        assert not isinstance(modellike, nn.Module), "modellike should not be a nn.Module!"
        self.modellike = modellike

    def init_modellike(self) -> nn.Module:
        """Initialize the modellike to be an actual model."""
        model = init_model_from_model_like(self.modellike)
        ModeloptStateManager.transfer_state_dict(self, model)
        return model


def _check_init_modellike(model: nn.Module, mode: _ModeDescriptor) -> nn.Module:
    """Utility to initialize a ModelLikeModule if needed according to the mode."""
    if mode.require_model_like:
        assert isinstance(model, ModelLikeModule), "Model must be a ModelLikeModule!"
    elif isinstance(model, ModelLikeModule):
        model = model.init_modellike()
    return model


def apply_mode(
    model: ModelLike,
    mode: ModeLike,
    registry: Optional[_ModeRegistryCls] = None,
    init_state: Optional[bool] = None,
) -> nn.Module:
    """Apply the provided modes the model, record the changes, and return the model.

    Args:
        model: A model-like object. Can be an nn.Module, a model class type, or a tuple.
            Tuple must be of the form ``(model_cls,)`` or ``(model_cls, args)`` or
            ``(model_cls, args, kwargs)``. Model will be initialized as
            ``model_cls(*args, **kwargs)``.
        mode: A mode, a list of modes or a list of tuples containing the mode and its config. The
            mode may be specified as a string or as the actual
            :mod:`_ModeDescriptor<modelopt.torch.opt.mode._ModeDescriptor>` class such as
            :mod:`QuantizeModeDescriptor<modelopt.torch.opt.quantization.QuantizeModeDescriptor>` class.
        registry: An optional mode registry from which to retrieve the mode. If not provided, all
            registries will be searched.
        init_state: Flag indicating whether we should initialize the state manager for the model. If
            not provided, it will be inferred from the model. This flag can be used to enforce a
            certain behavior. For example, for ``init_state=True`` the state manager will raise an
            error if the model already contains state.

    Returns:
        The converted model after applying the desired modes.
    """
    # initialize ModelLikeModule if needed.
    model = model if isinstance(model, nn.Module) else ModelLikeModule(model)

    # vanilla case (just initialize+return)
    if not mode:
        return model.init_modellike() if isinstance(model, ModelLikeModule) else model

    # check if the model is in a wrapper
    model = unwrap_model(model, raise_error=True)

    # standardize mode to ModeConfigList
    mode_and_config = get_mode_config(mode)

    # get or initialize the state manager for the model
    manager = ModeloptStateManager(
        model,
        init_state=(
            not ModeloptStateManager.is_converted(model) if init_state is None else init_state
        ),
    )

    # get mode function based on registry argument
    get_mode = registry.__getitem__ if registry else _ModeRegistryCls.get_from_any

    # update metadata of currently last mode before adding new modes
    manager.update_last_state_before_new_mode(model)

    # check whether a ModelLike should be initialized
    model = _check_init_modellike(model, get_mode(mode_and_config[0][0]))

    # loop through modes and call convert entrypoint for each mode and record data in manager.
    for m, config in mode_and_config:
        manager.check_mode(m)
        config = manager.get_config_class(m, config)
        model, metadata = get_mode(m).convert(model, config)
        manager.add_mode(m, config, metadata)

    # If the existing mode is empty, create an model instance from ModelLikeModule.
    if not manager.has_state and isinstance(model, ModelLikeModule):
        model = model.init_modellike()

    # it cannot be a ModelLikeModule anymore at the end
    assert not isinstance(model, ModelLikeModule), "Model must be a regular Module now!"

    # return model with state recorded
    return model


def get_mode(model: nn.Module) -> Optional[_ModeDescriptor]:
    """Get mode of converted network.

    model: A model that contains modelopt_state

    The mode of the model is defined as the last mode activated during the convert process.
    """
    if ModeloptStateManager.is_converted(model):
        return ModeloptStateManager(model).last_mode
    return None


def modelopt_state(model: nn.Module) -> Dict[str, Any]:
    """Return the modelopt state dict describing the modifications to the model.

    Note that the returned ``modelopt_state`` does not contain the model parameters such as weights and biases.
    ``modelopt_state`` is useful for saving and loading various modelopt optimization states separately from the
    model parameters. For example:

    .. code-block::

        import modelopt.torch.opt as mto

        # Save the modelopt state and model weights separately
        torch.save(mto.modelopt_state(model), "modelopt_state.pt") # Save the modelopt state
        torch.save(model.state_dict(), "model_weights.pt") # Save the model weights

    If you want to save the model weights and the modelopt state together, please use
    :meth:`mto.save()<modelopt.torch.opt.conversion.save>`.

    Args:
        model: the modelopt-modified model.

    Returns:
        An modelopt state dictionary describing the modifications to the model.
    """
    # unwrap model
    model = unwrap_model(model, warn=True)

    # retrieve state manager
    manager = ModeloptStateManager(
        model=model if ModeloptStateManager.is_converted(model) else None
    )

    # update metadata of current mode as needed
    manager.update_last_state_before_save(model)

    # construct state dict and return it
    objs = {
        "modelopt_state_dict": (
            manager.state_dict()
        ),  # empty state_dict is okay (saving regular models)
        "modelopt_version": __version__,
    }
    return objs


def save(model: nn.Module, f: Union[str, os.PathLike, BinaryIO], **kwargs) -> None:
    """Save a model's state dict together with the modelopt state dict to restore its architecture.

    Args:
        model: Any model.
        f: Target file location.
        **kwargs: additional args for ``torch.save()``.

    .. note::

        If model is a wrapper such as DistributedDataParallel, it will be unwrapped for saving.
    """
    # unwrap model
    model = unwrap_model(model, warn=True)

    # store ckpt
    ckpt_dict = {
        "modelopt_state": modelopt_state(model),
        "model_state_dict": model.state_dict(),
    }

    # store object
    torch.save(ckpt_dict, f, **kwargs)


def restore_from_modelopt_state(model: ModelLike, modelopt_state: Dict[str, Any]) -> nn.Module:
    """Restore the model architecture from the modelopt state dictionary based on the user-provided model.

    This method does not restore the model parameters such as weights and biases.
    Please load the weights and biases with the original checkpoint loading method after restoring
    modelopt states with `restore_from_modelopt_state`. For example:

    .. code-block:: python

        import modelopt.torch.opt as mto

        model = ...  # Create the model-like object

        # Restore the previously saved modelopt state followed by model weights
        mto.restore_from_modelopt_state(model, torch.load("modelopt_state.pt"))  # Restore modelopt state
        model.load_state_dict(torch.load("model_weights.pt"), ...)  # Load the model weights

    If you want to restore the model weights and the modelopt state together, please use
    :meth:`mto.restore()<modelopt.torch.opt.conversion.restore>`.

    Args:
        model: A model-like object. Can be an nn.Module, a model class type, or a tuple.
            Tuple must be of the form ``(model_cls,)`` or ``(model_cls, args)`` or
            ``(model_cls, args, kwargs)``. Model will be initialized as
            ``model_cls(*args, **kwargs)``.
        modelopt_state: The modelopt state dict describing the modelopt modifications to the model. The
            ``modelopt_state`` can be generated via
            :meth:`mto.modelopt_state()<modelopt.torch.opt.conversion.modelopt_state>`.

    Returns:
        A modified model architecture based on the restored modifications with the unmodified
        weights as stored in the provided ``model`` argument.

    .. note::

        Note that wrappers such as DistributedDataParallel are `not` supported during the restore
        process. Please wrap the model after the restore process.
    """
    # initialize ModelLikeModule if needed.
    model = model if isinstance(model, nn.Module) else ModelLikeModule(model)

    # Alert if the first 2 version numbers do not match, e.g., 0.3.2 vs 0.4.0.
    version = modelopt_state["modelopt_version"]
    if tuple(version.split(".")[:2]) != tuple(__version__.split(".")[:2]):
        warnings.warn(
            f"The checkpoint is stored with version {version}, but current version is"
            f" {__version__}. Compatibility of checkpoint with current version is not guaranteed!"
        )

    # initialize state manager and load state
    manager = ModeloptStateManager(model=model, init_state=True)
    manager.load_state_dict(modelopt_state["modelopt_state_dict"])

    # apply restore entrypoints for each of the modes
    for i, (m, config, metadata) in enumerate(manager.modes_with_states()):
        if i == 0:
            model = _check_init_modellike(model, m)
        model = m.restore(model, config, metadata)

    # If the existing mode is empty, create an model instance from ModelLikeModule.
    if not manager.has_state and isinstance(model, ModelLikeModule):
        model = model.init_modellike()

    # it cannot be a ModelLikeModule anymore at the end
    assert not isinstance(model, ModelLikeModule), "Model must be a regular Module now!"

    return model


def restore(model: ModelLike, f: Union[str, os.PathLike, BinaryIO], **kwargs) -> nn.Module:
    """Load the checkpoint, restore the modelopt model modifications, and load the model's weights.

    Args:
        model: A model-like object. Can be an nn.Module, a model class type, or a tuple.
            Tuple must be of the form ``(model_cls,)`` or ``(model_cls, args)`` or ``(model_cls, args, kwargs)``.
            Model will be initialized as ``model_cls(*args, **kwargs)``.
        f: Target file location generated by :meth:`mto.save()<modelopt.torch.opt.conversion.save>`.
        **kwargs: additional args for ``torch.load()``.

    Returns:
        The model with original weights and stored architecture.

    .. note::

        Note that wrappers such as DistributedDataParallel are `not` supported during the restore
        process. Please wrap the model after the restore process.
    """
    # initialize ModelLikeModule if needed.
    model = model if isinstance(model, nn.Module) else ModelLikeModule(model)

    # load checkpoint
    kwargs.setdefault("map_location", "cpu")
    objs = torch.load(f, **kwargs)

    # restore model architecture
    model_restored = restore_from_modelopt_state(model, objs["modelopt_state"])

    # load weights from checkpoint
    model_restored.load_state_dict(objs["model_state_dict"])

    # it cannot be a ModelLikeModule anymore at the end
    assert not isinstance(model_restored, ModelLikeModule), "Model must be a regular Module now!"

    return model_restored


# TODO: add a generic export function that will apply all remaining export modes.
def export(model: nn.Module) -> nn.Module:
    """Fully export the model to a regular model and finalize any model modifications."""
    raise NotImplementedError
