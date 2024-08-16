# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Configurations for distillation modes."""

from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import pydantic
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

from .loss_balancers import DistillationLossBalancer

__all__ = ["KDLossConfig"]

TeacherModel = Union[Type[nn.Module], Tuple, Callable]
Criterion = Union[Loss, Dict[Tuple[str, str], Loss]]


class KDLossConfig(ModeloptBaseConfig):
    """Configuration for the Knowledge-Distillation mode.

    This mode is used to distill knowledge from a teacher model to a student model.
    """

    # TODO: we should really think about a better to configure KDLossConfig
    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    teacher_model: Optional[TeacherModel] = ModeloptField(
        default=None,
        title="Teacher model",
        description=(
            "The class or callable or tuple to initialize the teacher model using"
            " :meth:`init_model_from_model_like"
            " <modelopt.torch.utils.network.init_model_from_model_like>`. This cannot already be an"
            " instance of nn.Module."
        ),
    )
    criterion: Optional[Criterion] = ModeloptField(
        default=None,
        title="Output-only or layer-wise loss criterion",
        description=(
            "If an instance of Loss class, a distillation loss will only be computed "
            "between outputs of a student and teacher; if a dictionary in the format "
            "{(student_layer_name, teacher_layer_name): loss_module}, a distillation loss will be "
            "computed for each specified student-teacher pair of layers using the corresponding "
            "``loss_module``."
        ),
    )
    loss_balancer: Optional[Any] = ModeloptField(
        default=None,
        title="Loss balancer",
        description=(
            "A balancer to reduce distillation and non-distillation losses into a single "
            "value using some weighing scheme."
        ),
    )
    expose_minimal_state_dict: bool = ModeloptField(
        default=True,
        title="Expose student state dict only",
        description=(
            "Hide teacher model's state_dict in the returned wrapped model. This reduces the "
            "checkpoint size by not re-storing the teacher unnecessarily again. "
            ".. note: Set to False if using `FSDP <https://pytorch.org/docs/stable/fsdp.html>`_"
        ),
    )

    @pydantic.field_validator("criterion")
    @classmethod
    def format_criterion(cls, criterion: Optional[Criterion]) -> Dict[Tuple[str, str], Loss]:
        """Ensure criterion is a mapping from layer names to loss (potentially entire module)."""
        if not isinstance(criterion, dict):
            # Output-only distillation.
            criterion = {("", ""): criterion}
        return criterion

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """Dump the config to a dictionary but avoid serializing teacher model to dict.

        This avoids issues when the teacher is a tuple with callable and args.
        If any of the args are Dataclasses, they are dumped as a dict and cannot be restored with their class.
        """
        return {**super().model_dump(*args, **kwargs), "teacher_model": self.teacher_model}

    def _strict_validate(self) -> None:
        """Strictly validate the configuration."""
        for k, v in self.items():
            if k == "loss_balancer":
                pydantic.TypeAdapter(
                    Optional[DistillationLossBalancer], config=self.model_config
                ).validate_python(v)
                continue
            assert v is not None, f"Missing required field: {k}."

        # Cannot have multiple loss layers without LossBalancer.
        if self.loss_balancer is not None:
            assert (
                len(tuple(self.loss_balancer.parameters())) == 0
            ), "Loss Balancer cannot have parameters."
        else:
            assert (
                len(self.criterion) == 1  # type: ignore[arg-type]
            ), "Cannot have multiple layer-loss pairs without a `DistillationLossBalancer`"


class ExportStudentConfig(ModeloptBaseConfig):
    """Configuration for the export_student mode.

    This mode is used to export a student model after distillation training/fine-tuning.
    """
