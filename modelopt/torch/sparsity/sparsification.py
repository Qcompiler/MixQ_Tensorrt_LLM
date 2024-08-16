# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""High-level API to automatically sparsify your model with various algorithms."""
from typing import Any, Dict, Optional, Tuple, Type

from torch import nn

from modelopt.torch.opt.conversion import apply_mode, get_mode
from modelopt.torch.opt.mode import ModeLike
from modelopt.torch.opt.searcher import BaseSearcher, SearchConfig
from modelopt.torch.utils import unwrap_model

from .mode import SparsityModeRegistry

__all__ = ["sparsify", "export"]


def sparsify(
    model: nn.Module, mode: ModeLike, config: Optional[SearchConfig] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Sparsify a given model and search for they optimal sparsified weights.

    Args:
        model: A standard model that contains standard building blocks to be sparsified in-place.
        mode: A (list of) string(s) or Mode(s) or a list of tuples containing the mode and its
            config indicating the desired mode(s) (and configurations) for the convert
            process. Modes set up the model for different algorithms for model optimization. The
            following modes are available:

            *   :meth:`"sparse_magnitude"<modelopt.torch.sparsity.mode.SparseMagnitudeModeDescriptor>`:
                The ``model`` will be sparsified according to the magnitude of weights in each
                layer. The mode's config is described in
                :meth:`SparseMagnitudeConfig<modelopt.torch.sparsity.config.SparseMagnitudeConfig>`.
            *   :meth:`"sparsegpt"<modelopt.torch.sparsity.mode.SparseGPTModeDescriptor>`:
                The ``model`` will be sparsified and weights are updated optimally using an Hessian
                approximation of the loss function (see SparseGPT paper for details). The mode's
                config is described in
                :meth:`SparseGPTConfig<modelopt.torch.sparsity.config.SparseGPTConfig>`.

            If the mode argument is specified as a dictionary, the keys should indicate the mode and
            the values specify the per-mode configuration. If not provided, then default
            configuration would be used.

        config: Additional optional arguments to configure the search. Currently, we support:

            * ``verbose``: Whether to print detailed search stats during search.
            * ``forward_loop``: A ``Callable`` that takes a model as input and runs a forward loop
                on it. It is recommended to choose the data loader used inside the forward loop
                carefully to reduce the runtime. Cannot be provided at the same time as
                ``data_loader`` and ``collect_func``.
            * ``data_loader``: An iterator yielding batches of data for calibrating the
              normalization layers in the model or compute gradient scores. It is recommended to use
              the same data loader as for training but with significantly fewer iterations. Cannot
              be provided at the same time as ``forward_loop``.
            * ``collect_func``: A ``Callable`` that takes a batch of data from the data loader as
              input and returns the input to ``model.forward()`` as described in
              :meth:`run_forward_loop <modelopt.torch.utils.network.run_forward_loop>`. Cannot
              be provided at the same time as ``forward_loop``.

            .. note::

                Additional configuration options may be added by individual algorithms. Please
                refer to the documentation of the individual algorithms for more information.

    Returns: A sparsified model

    .. note::

        The given model is sparsified in-place. The returned model is thus a reference to the same
        model instance as the input model.
    """
    # apply sparsity to the model
    model = apply_mode(model, mode, registry=SparsityModeRegistry)

    # retrieve searcher class
    searcher_cls: Type[BaseSearcher] = getattr(get_mode(model), "search_algorithm")

    # run search+sparsification algorithm
    searcher = searcher_cls()
    searcher.search(model, {}, tuple(), config)

    # return the sparsified model
    return model


def export(model: nn.Module) -> nn.Module:
    """Export a sparse dynamic model to a regular model.

    This should be done after the model is fine-tuned and the weights are fixed.

    .. warning::

        After the call to ``export()``, the sparsity mask will no longer be enforced. This means any
        future weight updates would destroy the sparsity pattern. If you want to continue training,
        call ``export()`` after training is finished.
    """
    # unwrap a DP/DDP model
    model = unwrap_model(
        model,
        warn=True,
        msg=(
            f"Unwrapping a {type(model).__name__} model for export! Note that the export is"
            " in-place and the model wrapper should be re-created after export since the wrapper"
            " might not support changing parameters after initialization."
        ),
    )

    # apply export mode and return model
    return apply_mode(model, "export_sparse", registry=SparsityModeRegistry)
