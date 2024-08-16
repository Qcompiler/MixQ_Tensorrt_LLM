# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


"""Utility functions for using torch.distributed."""

import functools
import io
import os
import time
from typing import Any, List, Optional, Union

import torch
import torch.distributed
from packaging.version import Version

__all__ = [
    "backend",
    "barrier",
    "get_data_parallel_group",
    "get_tensor_parallel_group",
    "is_available",
    "is_initialized",
    "is_master",
    "rank",
    "set_data_parallel_group",
    "set_tensor_parallel_group",
    "size",
]


def is_available() -> bool:
    """Returns whether the distributed package is available."""
    return torch.distributed.is_available()


def is_initialized() -> bool:
    """Returns whether the distributed package is initialized."""
    return is_available() and torch.distributed.is_initialized()


def backend() -> Optional[str]:
    """Returns the distributed backend."""
    if is_initialized():
        return "torch"
    return None


def size(group=None) -> int:
    """Returns the number of processes."""
    if backend() == "torch":
        return torch.distributed.get_world_size(group=group)
    return 1


def rank(group=None) -> int:
    """Returns the rank of the current process."""
    if backend() == "torch":
        return torch.distributed.get_rank(group=group)
    return 0


def is_master(group=None) -> bool:
    """Returns whether the current process is the master process."""
    return rank(group=group) == 0


def _serialize(obj: Any) -> torch.Tensor:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    if Version(torch.__version__) >= Version("1.13"):
        storage = torch.UntypedStorage.from_buffer(buffer.getvalue(), dtype=torch.uint8)
    else:
        storage = torch._UntypedStorage.from_buffer(buffer.getvalue(), dtype=torch.uint8)
    tensor = torch.ByteTensor(storage)
    return tensor


def _deserialize(tensor: torch.Tensor, size: Optional[int] = None) -> Any:
    buffer = tensor.numpy().tobytes()
    if size is not None:
        buffer = buffer[:size]
    obj = torch.load(io.BytesIO(buffer))
    return obj


def _broadcast(tensor: torch.Tensor, src: int = 0, group=None) -> None:
    if backend() == "torch":
        torch.distributed.broadcast(tensor, src, group)


def broadcast(obj: Any, src: int = 0, group=None) -> Any:
    """Broadcasts an object from the source to all other processes."""
    if size() == 1:
        return obj

    # serialize
    if rank() == src:
        tensor = _serialize(obj).cuda()

    # broadcast the tensor size
    if rank() == src:
        tensor_size = torch.LongTensor([tensor.numel()]).cuda()
    else:
        tensor_size = torch.LongTensor([0]).cuda()
    _broadcast(tensor_size, src=src, group=group)

    # broadcast the tensor
    if rank() != src:
        tensor = torch.ByteTensor(size=(tensor_size.item(),)).cuda()
    _broadcast(tensor, src=src, group=group)

    # deserialize
    if rank() != src:
        obj = _deserialize(tensor.cpu())
    return obj


def _allgather(tensors: List[torch.Tensor], tensor: torch.Tensor) -> None:
    if backend() == "torch":
        torch.distributed.all_gather(tensors, tensor)


def allgather(obj: Any) -> List[Any]:
    """Gathers an object from all processes into a list."""
    if size() == 1:
        return [obj]

    # serialize
    tensor = _serialize(obj).cuda()

    # gather the tensor size
    tensor_size = torch.LongTensor([tensor.numel()]).cuda()
    tensor_sizes = [torch.LongTensor([0]).cuda() for _ in range(size())]
    _allgather(tensor_sizes, tensor_size)
    tensor_sizes = [int(tensor_size.item()) for tensor_size in tensor_sizes]
    max_size = max(tensor_sizes)

    # gather the tensor
    tensors = [torch.ByteTensor(size=(max_size,)).cuda() for _ in tensor_sizes]
    if tensor_size != max_size:
        padding = torch.ByteTensor(size=(max_size - tensor_size,)).cuda()
        tensor = torch.cat((tensor, padding), dim=0)
    _allgather(tensors, tensor)

    # deserialize
    objs = []
    for tensor_size, tensor in zip(tensor_sizes, tensors):
        obj = _deserialize(tensor.cpu(), size=tensor_size)
        objs.append(obj)
    return objs


def allreduce(obj: Any, reduction: str = "sum") -> Any:
    """Reduces an object from all processes."""
    objs = allgather(obj)
    if reduction == "sum":
        return sum(objs)
    else:
        raise NotImplementedError(reduction)


def barrier(group=None) -> None:
    """Synchronizes all processes."""
    if size() == 1:
        return
    if backend() == "torch":
        torch.distributed.barrier(group=group)


def master_only(func):
    """Decorator to run a function only on the master process and broadcast the result."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return broadcast(func(*args, **kwargs) if is_master() else None)

    return wrapper


class DistributedProcessGroup:
    """A class to manage the distributed process group for data parallel, tensor parallel etc."""

    def __init__(self, group=None):
        """Initialize the distributed process group."""
        self.group = group

    def is_initialized(self) -> bool:
        """Check if the distributed process group is initialized."""
        return backend() == "torch" and self.group != -1

    def rank(self) -> int:
        """Get the rank of the current process group."""
        return rank(group=self.group) if self.is_initialized() else -1

    def world_size(self) -> int:
        """Get the world size of the current process group."""
        return size(group=self.group) if self.is_initialized() else -1


_DATA_PARALLEL_GROUP = DistributedProcessGroup(None)
_TENSOR_PARALLEL_GROUP = DistributedProcessGroup(-1)


def set_data_parallel_group(group: Optional[Union[torch.distributed.ProcessGroup, int]]):
    """Set the data parallel group."""
    _DATA_PARALLEL_GROUP.group = group


def set_tensor_parallel_group(group: Optional[Union[torch.distributed.ProcessGroup, int]]):
    """Set the tensor parallel group."""
    _TENSOR_PARALLEL_GROUP.group = group


def get_data_parallel_group() -> DistributedProcessGroup:
    """Get the data parallel group."""
    return _DATA_PARALLEL_GROUP


def get_tensor_parallel_group() -> DistributedProcessGroup:
    """Get the tensor parallel group."""
    return _TENSOR_PARALLEL_GROUP


def get_group(ranks: List[int]):
    """Returns the process group if torch.distributed.is_initialized()."""
    # NCCL has an issue with calling barrier. So we just use the gloo backebnd for group barriers.
    return torch.distributed.new_group(ranks, backend="gloo") if is_initialized() else None


class FileLock:
    """Mutex object for writing to a file atomically using the O_EXCL directive on Unix filesystems."""

    def __init__(
        self,
        lockfile_path: str,
        all_acquire: bool = False,
        poll_time: float = 0.25,
    ):
        """Constructor.

        Args:
            lockfile_path: Path to a nonexistent file to be used as the locking mechanism.
            all_acquire: Will keep retrying to acquire a lock if True.
            poll_time: Sleep interval between retries.
        """
        self.lockfile_path = lockfile_path
        self.all_acquire = all_acquire
        self.poll_time = poll_time
        self.handle = None

    def try_acquire(self):  # noqa: D102
        try:
            self.handle = os.open(self.lockfile_path, os.O_CREAT | os.O_EXCL)
            return True
        except FileExistsError:
            return False

    def wait(self):  # noqa: D102
        while os.path.exists(self.lockfile_path):
            time.sleep(self.poll_time)

    def release(self):  # noqa: D102
        if self.handle is not None:
            os.close(self.handle)
        os.remove(self.lockfile_path)

    def __enter__(self):
        while True:
            if self.try_acquire() or not self.all_acquire:
                break
            self.wait()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
