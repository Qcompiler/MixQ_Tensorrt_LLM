# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Utility functions for performance measurement."""

from contextlib import ContextDecorator

import torch

from . import distributed as dist
from .logging import print_rank_0

__all__ = ["clear_cuda_cache", "get_cuda_memory_stats", "report_memory", "Timer"]


def clear_cuda_cache():
    """Clear the CUDA cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_cuda_memory_stats(device=None):
    """Get memory usage of specified GPU in Bytes."""
    return {
        "allocated": torch.cuda.memory_allocated(device),
        "max_allocated": torch.cuda.max_memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "max_reserved": torch.cuda.max_memory_reserved(device),
    }


def report_memory(name="", rank=0):
    """Simple GPU memory report."""
    memory_stats = get_cuda_memory_stats()
    string = name + " memory (MB)"
    for k, v in memory_stats.items():
        string += f" | {k}: {v / 2**20: .2e}"

    if dist.is_initialized():
        string = f"[Rank {dist.rank()}] " + string
        if dist.rank() == rank:
            print(string, flush=True)
    else:
        print(string, flush=True)


class Timer(ContextDecorator):
    """A Timer that can be used as a decorator as well."""

    def __init__(self, name=""):
        """Initialize Timer."""
        super().__init__()
        self.name = name
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._stop_event = torch.cuda.Event(enable_timing=True)
        self.estimated_time = 0
        self.start()

    def start(self):
        """Start the timer."""
        self._start_event.record()
        return self

    def stop(self) -> float:
        """End the timer."""
        self._stop_event.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        self.estimated_time = self._start_event.elapsed_time(self._stop_event)
        return self.estimated_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()
        print_rank_0(f"{self.name} took {self.estimated_time:.3e} ms")
