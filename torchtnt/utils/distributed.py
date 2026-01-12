#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps
from multiprocessing.managers import SyncManager
from typing import Any, Callable, cast, Dict, Generator, List, Optional, TypeVar, Union

import torch
import torch.nn.functional as F
from torch import distributed as dist, multiprocessing, Tensor
from torch.distributed.elastic.utils.distributed import get_free_port
from typing_extensions import Literal, ParamSpec


T = TypeVar("T")
DistObjList = Union[List[T], List[None]]
TParams = ParamSpec("TParams")
TReturn = TypeVar("TReturn")

logger: logging.Logger = logging.getLogger(__name__)


class PGWrapper:
    """
    A wrapper around ProcessGroup that allows collectives to be issued in a
    consistent fashion regardless of the following scenarios:

        pg is None, distributed is initialized:     use WORLD as pg
        pg is None, distributed is not initialized: single process app
        pg is not None:                             use pg
    """

    def __init__(self, pg: Optional[dist.ProcessGroup]) -> None:
        if pg is None and dist.is_available() and dist.is_initialized():
            self.pg: Optional[dist.ProcessGroup] = dist.group.WORLD
        else:
            self.pg: Optional[dist.ProcessGroup] = pg

    def get_rank(self) -> int:
        if self.pg is None:
            return 0
        return dist.get_rank(group=self.pg)

    def get_world_size(self) -> int:
        if self.pg is None:
            return 1
        return dist.get_world_size(group=self.pg)

    def barrier(self) -> None:
        if self.pg is None:
            return
        backend = dist.get_backend(group=self.pg)
        if backend == dist.Backend.NCCL:
            dist.barrier(group=self.pg, device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier(group=self.pg)

    def broadcast_object_list(self, obj_list: DistObjList, src: int = 0) -> None:
        if self.pg is None:
            return
        dist.broadcast_object_list(obj_list, src=src, group=self.pg)

    def all_gather_object(self, obj_list: DistObjList, obj: T) -> None:
        if self.pg is None:
            obj_list = cast(List[T], obj_list)  # to make pyre happy
            obj_list[0] = obj
            return
        dist.all_gather_object(obj_list, obj, group=self.pg)

    def scatter_object_list(
        self,
        output_list: List[None],
        input_list: Optional[DistObjList],
        src: int = 0,
    ) -> None:
        rank = self.get_rank()
        world_size = self.get_world_size()
        if rank == src:
            if input_list is None:
                raise RuntimeError(
                    "The src rank's input_list for scatter_object_list must not be None."
                )
            if len(input_list) != world_size:
                raise RuntimeError(
                    f"The length of input_list {len(input_list)} for scatter_object_list "
                    f"must be the same as the process group's world size ({world_size})."
                )
        else:
            input_list = [None] * world_size

        if self.pg is None:
            output_list[0] = input_list[0]
            return

        # scatter_object_list does not yet support NCCL backend
        if dist.get_backend(self.pg) == "nccl":
            self.broadcast_object_list(obj_list=input_list, src=src)
            output_list[0] = input_list[rank]
            return

        dist.scatter_object_list(output_list, input_list, src=src, group=self.pg)


def get_global_rank() -> int:
    """
    Get rank using torch.distributed if available. Otherwise, the RANK env var instead if initialized.
    Returns 0 if neither condition is met.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()

    environ_rank = os.environ.get("RANK", "")
    if environ_rank.isdecimal():
        return int(os.environ["RANK"])

    return 0


def get_local_rank() -> int:
    """
    Get rank using the ``LOCAL_RANK`` environment variable, if populated: https://pytorch.org/docs/stable/elastic/run.html#environment-variables
    Defaults to 0 if ``LOCAL_RANK`` is not set.
    """
    environ_local_rank = os.environ.get("LOCAL_RANK")
    if environ_local_rank:
        return int(environ_local_rank)
    return 0


def get_local_world_size() -> int:
    """
    Get local world size using the ``LOCAL_WORLD_SIZE`` environment variable, if populated: https://pytorch.org/docs/stable/elastic/run.html#environment-variables
    Defaults to 1 if ``LOCAL_WORLD_SIZE`` is not set.
    """
    environ_local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if environ_local_world_size:
        return int(environ_local_world_size)
    return 1


def get_world_size() -> int:
    """
    Get world size using torch.distributed if available. Otherwise, the WORLD_SIZE env var is used instead if initialized.
    Returns 1 if neither condition is met.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()

    world_size = os.environ.get("WORLD_SIZE", "")
    if world_size.isdecimal():
        return int(world_size)

    return 1


def barrier() -> None:
    """
    Add a synchronization point across all processes when using distributed.
    If torch.distributed is initialized, this function will invoke a barrier across the global process group.
    For more granular process group wrapping, please refer to :class:`~torchtnt.utils.PGWrapper`.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    backend = dist.get_backend()
    if backend == dist.Backend.NCCL:
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def destroy_process_group() -> None:
    """Destroy the global process group, if one is already initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_process_group_backend_from_device(device: torch.device) -> str:
    """Function that gets the default process group backend from the device."""
    return "nccl" if device.type == "cuda" else "gloo"


def _validate_global_rank_world_size(world_size: int, rank: int) -> None:
    if world_size < 1:
        raise ValueError(
            f"Invalid world_size value provided: {world_size}. Value must be greater than 0."
        )
    if rank < 0:
        raise ValueError(
            f"Invalid rank value provided: {rank}. Value must be greater than non-negative."
        )
    if rank >= world_size:
        raise ValueError(
            f"Invalid rank and world_size values provided: rank={rank}, world_size={world_size}. Rank must be less than world_size."
        )


def get_file_init_method(
    *,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    filename: Optional[str] = None,
) -> str:
    """Gets init method for the TCP protocol for the distributed environment.
    For more information, see here: https://pytorch.org/docs/stable/distributed.html#shared-file-system-initialization

    Args:
        world_size: global number of workers. If ``None``, the default is fetched using :function:`get_world_size`.
        rank: Global rank of the worker calling the function. If ``None``, the default is fetched using :function:`get_global_rank`.
        filename: The filename to use for synchronization. If ``None``, a new temporary file is used.
    """
    world_size = world_size if world_size is not None else get_world_size()
    rank = rank if rank is not None else get_global_rank()
    _validate_global_rank_world_size(world_size, rank)
    if filename is None:
        with tempfile.NamedTemporaryFile() as tmp_file:
            filename = tmp_file.name
    init_method = f"file://{filename}?world_size={world_size}&rank={rank}"
    return init_method


def get_tcp_init_method(
    *,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    hostname: Optional[str] = None,
    port: Optional[int] = None,
) -> str:
    """Gets init method for the TCP protocol for the distributed environment.
    For more information, see here: https://pytorch.org/docs/stable/distributed.html#tcp-initialization.

    Args:
        world_size: global number of workers. If ``None``, the default is fetched using :function:`get_world_size`.
        rank: Global rank of the worker calling the function. If ``None``, the default is fetched using :function:`get_global_rank`.
        hostname: an address that belongs to the rank 0 process. If ``None``, then ``localhost`` is used.
        port: A free port to use for communication. If ``None``, this port is automatically selected.
    """
    world_size = world_size if world_size is not None else get_world_size()
    rank = rank if rank is not None else get_global_rank()
    _validate_global_rank_world_size(world_size, rank)
    host_addr = hostname if hostname is not None else "localhost"
    host_port = port if port is not None else get_free_port()
    init_method = f"tcp://{host_addr}:{host_port}?world_size={world_size}&rank={rank}"
    return init_method


def _simple_all_gather_tensors(
    result: Tensor, group: Optional[dist.ProcessGroup], world_size: int
) -> List[Tensor]:
    stacked_result_sizes = [world_size] + list(result.size())
    gathered_result = list(
        torch.zeros(stacked_result_sizes, dtype=result.dtype, device=result.device)
    )
    dist.all_gather(gathered_result, result, group)
    return gathered_result


def all_gather_tensors(
    result: Tensor, group: Optional[dist.ProcessGroup] = None
) -> List[Tensor]:
    """Function to gather tensors from several distributed processes onto a list that is broadcasted to all processes.
    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        gathered_result: list with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    # if torch.distributed is not available or not initialized
    # return single-item list containing the result
    if not dist.is_available() or not dist.is_initialized():
        return [result]

    # convert tensors to contiguous format
    result = result.contiguous()
    world_size = dist.get_world_size(group)

    # if the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_all_gather_tensors(result, group, world_size)

    # gather sizes of all tensors
    local_size = torch.tensor(result.shape, device=result.device)
    stacked_local_size = [world_size] + list(local_size.size())
    local_sizes = list(
        torch.zeros(
            stacked_local_size, dtype=local_size.dtype, device=local_size.device
        )
    )
    dist.all_gather(local_sizes, local_size, group=group)

    # if the backend is NCCL, we can gather the differently sized tensors without padding
    if dist.get_backend(group) == "nccl":
        gathered_result = [result.new_empty(size.tolist()) for size in local_sizes]
        dist.all_gather(gathered_result, result, group)
        return gathered_result

    # if shapes are all the same, then do a simple gather:
    stacked_sizes = torch.stack(local_sizes)
    max_size = stacked_sizes.max(dim=0).values
    min_size = stacked_sizes.min(dim=0).values
    all_sizes_equal = torch.equal(max_size, min_size)
    if all_sizes_equal:
        return _simple_all_gather_tensors(result, group, world_size)

    # if not, we need to pad each local tensor to maximum size, gather and then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    stacked_result_padded = [world_size] + list(result_padded.size())
    gathered_result = list(
        torch.zeros(
            stacked_result_padded,
            dtype=result_padded.dtype,
            device=result_padded.device,
        )
    )
    dist.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result


TReturn = TypeVar("TReturn")


def rank_zero_fn(
    fn: Callable[TParams, TReturn],
) -> Callable[TParams, Optional[TReturn]]:
    """Function that can be used as a decorator to enable a function to be called on global rank 0 only.

    Note:
        This decorator should be used judiciously. it should never be used on functions that need synchronization.
        It should be used very carefully with functions that mutate local state as well

    Example:

        >>> from torchtnt.utilities.distributed import rank_zero_fn
        >>> @rank_zero_fn
        ... def foo():
        ...     return 1
        ...
        >>> x = foo() # x is 1 if global rank is 0 else x is None

    Args:
        fn: the desired function to be executed on rank 0 only

    Return:
        wrapped_fn: the wrapped function that executes only if the global rank is  0

    """

    @wraps(fn)
    def wrapped_fn(*args: TParams.args, **kwargs: TParams.kwargs) -> Optional[TReturn]:
        if get_global_rank() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


def local_rank_zero_fn(
    fn: Callable[TParams, TReturn],
) -> Callable[TParams, Optional[TReturn]]:
    """Function that can be used as a decorator to enable a function to be called on local rank 0 only.

    Note:
        This decorator should be used judiciously. it should never be used on functions that need synchronization.
        It should be used very carefully with functions that mutate local state as well

    Example:

        >>> from torchtnt.utilities.distributed import local_rank_zero_fn
        >>> @local_rank_zero_fn
        ... def foo():
        ...     return 1
        ...
        >>> x = foo() # x is 1 if local rank is 0 else x is None

    Args:
        fn: the desired function to be executed on rank 0 only

    Return:
        wrapped_fn: the wrapped function that executes only if the global rank is  0

    """

    @wraps(fn)
    def wrapped_fn(*args: TParams.args, **kwargs: TParams.kwargs) -> Optional[TReturn]:
        if get_local_rank() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


class _BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    """
    The only difference between :class:`torch.nn.BatchNorm1d`, :class:`torch.nn.BatchNorm2d`,
    :class:`torch.nn.BatchNorm3d`, etc is this method that is overwritten by the sub-class.
    This method is used when calling forward as a sanity check.
    When using :function:`revert_sync_batchnorm` this sanity check is lost.
    """

    def _check_input_dim(self, input: Tensor) -> None:
        return


def revert_sync_batchnorm(
    module: torch.nn.Module, device: Optional[Union[str, torch.device]] = None
) -> torch.nn.Module:
    """
    Helper function to convert all :class:`torch.nn.SyncBatchNorm` layers in the module to
    :attr:`BatchNorm*D` layers. This function reverts :meth:`torch.nn.SyncBatchNorm.convert_sync_batchnorm`.

    Args:
        module (nn.Module): module containing one or more :class:`torch.nn.SyncBatchNorm` layers
        device (optional): device in which the :attr:`BatchNorm*D` should be created,
                default is cpu

    Returns:
        The original :attr:`module` with the converted :attr:`BatchNorm*D`
        layers. If the original :attr:`module` is a :class:`torch.nn.SyncBatchNorm` layer,
        a new :attr:`BatchNorm*D` layer object will be returned
        instead. Note that the :attr:`BatchNorm*D` layers returned will not have input dimension information.

    Example::

        >>> # Network with nn.BatchNorm layer
        >>> module = torch.nn.Sequential(
        >>>            torch.nn.Linear(20, 100),
        >>>            torch.nn.BatchNorm1d(100),
        >>>          ).cuda()
        >>> sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
        >>> reverted_module = revert_sync_batchnorm(sync_bn_module, torch.device("cuda"))

    """
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        module_output = _BatchNormXd(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
            device,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            # pyre-fixme[16]: `_BatchNormXd` has no attribute `qconfig`.
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child, device))
    del module
    return module_output


def sync_bool(
    val: bool,
    pg: Optional[dist.ProcessGroup] = None,
    coherence_mode: Union[Literal["any", "all", "rank_zero"], int, float] = "any",
) -> bool:
    """Utility to synchronize a boolean value across members of a provided process group.

    In the case ``torch.distributed`` is not available or initialized, the input ``val`` is returned.

    Args:
        val (bool): boolean value to synchronize
        pg: process group to use for synchronization. If not specified, the default process group is used.
        coherence_mode Union[str, int, float]: the manner in which the boolean value should be synchronized. 5 options are currently supported:
            1. any (default): If any rank provides a True value, all ranks should receive True.
            2. all: Only if all ranks provide a True value should all ranks receive True.
            3. rank_zero: Makes rank 0 process's value the source of truth and broadcasts the result to all other processes.
            4. If an integer N is provided, return True only if at least N processes provide a True value.
            5. If a float F is provided, return True only if at least this ratio of processes provide a True value. The ratio provided should be in the range [0, 1].

    Returns:
        The synchronized boolean value.

    Example::

        >>> val = True
        >>> # synced_val is True iff all ranks provide a True value to the function
        >>> synced_val = sync_bool(val, coherence_mode="all")
        >>> if synced_val:
        >>>     print("success")

    """
    if not dist.is_available() or not dist.is_initialized():
        return val

    pg = pg or dist.group.WORLD
    device = torch.device(
        torch.cuda.current_device() if dist.get_backend(pg) == "nccl" else "cpu"
    )
    pg_wrapper = PGWrapper(pg)

    dtype = torch.uint8
    if pg_wrapper.get_world_size() > 256:
        dtype = torch.int

    indicator = (
        torch.ones(1, device=device, dtype=dtype)
        if val
        else torch.zeros(1, device=device, dtype=dtype)
    )

    if coherence_mode == "rank_zero":
        # Broadcast from rank 0 to all other ranks
        dist.broadcast(indicator, src=0, group=pg)
        return bool(indicator[0].item())
    elif coherence_mode == "any":
        # sum up the indicators across all the ranks.
        dist.all_reduce(indicator, op=dist.ReduceOp.SUM)
        return indicator.item() > 0
    elif coherence_mode == "all":
        dist.all_reduce(indicator, op=dist.ReduceOp.SUM)
        return indicator.item() == pg_wrapper.get_world_size()
    elif isinstance(coherence_mode, int):
        # if >= int(coherence_mode) processes signal to stop, all processes stop
        dist.all_reduce(indicator, op=dist.ReduceOp.SUM)
        return indicator.item() >= coherence_mode
    elif isinstance(coherence_mode, float):
        dist.all_reduce(indicator, op=dist.ReduceOp.SUM)
        return (indicator.item() / pg_wrapper.get_world_size()) >= coherence_mode
    else:
        raise TypeError(
            f'Invalid value for `coherence_mode` provided: Expected type int, float, or one of ("any", "all", "rank_zero"), but received {coherence_mode}.'
        )


@dataclass
class ProcessGroupSetupParams:
    backend: str
    port: str
    world_size: int
    timeout_s: int


def spawn_multi_process(
    world_size: int,
    backend: str,
    method: Callable[TParams, TReturn],
    *method_args: Any,
    **method_kwargs: Any,
) -> List[TReturn]:
    """
    Spawn single node, multi-rank function.
    Uses localhost and free port to communicate.

    Args:
        world_size: number of processes
        backend: backend to use. for example, "nccl", "gloo", etc
        method: callable to spawn.
        method_args: args for the method
        method_kwargs: kwargs for the method

    Note:
        The default timeout used for distributed collectives in the process group is 60 seconds.
        This can be overridden by passing a `timeout_s` key in the `method_kwargs`. It will be
        extracted before passing to the method call.

    Returns:
        A list, l, where l[i] is the return value of method(*method_args, **methods_kwargs) on rank i
    """
    mp_init_mode = "spawn" if torch.version.hip is not None else None
    manager = SyncManager(ctx=multiprocessing.get_context(mp_init_mode))
    manager.start()
    mp_output_dict = manager.dict()

    port = str(get_free_port())
    torch.multiprocessing.spawn(
        # torch.multiprocessing.spawn sends rank as the first param
        # https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn
        _init_pg_and_rank_and_launch_method,
        args=(
            ProcessGroupSetupParams(
                backend=backend,
                port=port,
                world_size=world_size,
                timeout_s=method_kwargs.pop("timeout_s", 60),
            ),
            mp_output_dict,
            method,
            method_args,
            method_kwargs,
        ),
        nprocs=world_size,
    )

    output_list = []
    for i in range(world_size):
        output_list.append(mp_output_dict[i])

    return output_list


def _init_pg_and_rank_and_launch_method(
    rank: int,
    pg_setup_params: ProcessGroupSetupParams,
    mp_output_dict: Dict[int, object],
    method: Callable[TParams, TReturn],
    args: List[object],
    kwargs: Dict[str, object],
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = pg_setup_params.port
    os.environ["WORLD_SIZE"] = str(pg_setup_params.world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        rank=rank,
        world_size=pg_setup_params.world_size,
        backend=pg_setup_params.backend,
        timeout=timedelta(  # setting up timeout for distributed collectives
            seconds=pg_setup_params.timeout_s
        ),
    )
    try:
        # pyre-ignore: spawn_multi_process uses unsafe types to begin with
        mp_output_dict[rank] = method(*args, **kwargs)

    finally:
        destroy_process_group()


def rank_zero_read_and_broadcast(
    func: Callable[TParams, TReturn],
) -> Callable[TParams, TReturn]:
    """
    Decorator that ensures a function is only executed by rank 0 and returns the result to all ranks.

    Note:
        By default will use the global process group. To use a custom process group, `process_group` must be an arg to the function and passed as a keyword argument.
    """

    @wraps(func)
    def wrapper(*args: TParams.args, **kwargs: TParams.kwargs) -> TReturn:
        ret = None
        rank = get_global_rank()
        process_group = kwargs.pop("process_group", None)  # pyre-ignore[16]

        # Do all filesystem reads from rank 0 only
        if rank == 0:
            ret = func(*args, **kwargs)

        # If not running in a distributed setting, return as is
        if not (dist.is_available() and dist.is_initialized()):
            # we cast here to avoid type errors, since it is
            # guaranteed the return value is of type T
            return cast(TReturn, ret)

        # Otherwise, broadcast result from rank 0 to all ranks
        pg = PGWrapper(process_group)
        path_container = [ret]
        pg.broadcast_object_list(path_container, 0)
        val = path_container[0]

        # we cast here to avoid type errors, since it is
        # guaranteed the return value is of type T
        return cast(TReturn, val)

    return wrapper


def broadcast_str(
    val: Optional[str],
    src: int = 0,
    process_group: Optional[dist.ProcessGroup] = None,
    fixed_buffer_size: Optional[int] = None,
) -> Optional[str]:
    """
    Broadcasts a string from a source rank to all other ranks in a process group.
    Serializes string as sequence of uint8 and broadcasts as a tensor. This avoids
    issues with broadcast_object_list and related apis which use pickle to serialize objects.

    Args:
        val: the string to broadcast
        src: the source rank to broadcast from
        process_group: the process group to broadcast in. Defaults to the WORLD process group.
        fixed_buffer_size (int, optional): The fixed buffer size to use. Defaults to none.
            If provided, it reduces the number of collective calls by padding the string to a fixed length.

    Returns:
        The broadcasted string.

    Note:
        This function issues two collective calls, one to broadcast the size of the serialized string and
        one to broadcast the string itself. If you want to avoid two collective calls, you can pass a fixed_buffer_size
        parameter. This will cause the string to be padded to the fixed length and only one broadcast will be performed.
        However, this comes with the cost of extra memory usage.
        If the string length is less than the buffer size, src rank will terminate early. However, receiving ranks may see collective hang, as expecting data from src rank. Please ensure the buffer size is large enough to avoid this issue.
    """
    if not dist.is_available() or not dist.is_initialized():
        return val
    if fixed_buffer_size is not None and fixed_buffer_size <= 0:
        raise ValueError(f"Expected fixed_buffer_size > 0, got {fixed_buffer_size}")

    rank = dist.get_rank(group=process_group)

    # device to use when broadcasting the string
    device = torch.device(
        torch.cuda.current_device()
        if dist.get_backend(process_group) == "nccl"
        else "cpu"
    )

    # Initialize buffer and buffer_length for all ranks
    buffer = torch.empty((1), dtype=torch.uint8)
    buffer_length = torch.empty((1), dtype=torch.int, device=device)

    if rank == src:
        assert (
            val is not None
        ), "Source rank must provide a string to broadcast, got None"

        # convert string to tensor
        buffer = torch.frombuffer(val.encode("utf-8"), dtype=torch.uint8)
        buffer = buffer.to(device=device)
        buffer_length = torch.tensor([len(buffer)], dtype=torch.int, device=device)

        if fixed_buffer_size is not None:
            if len(buffer) > fixed_buffer_size:
                raise ValueError(
                    f"Serialized string size ({len(buffer)}) exceeds buffer size ({fixed_buffer_size})"
                )
            # Pad the buffer with a special value (e.g., 0) to indicate the end of the string
            buffer = F.pad(buffer, (0, fixed_buffer_size - len(buffer)), value=0)

    # first broadcast the buffer length so receiving ranks can allocate the correct amount of memory
    if fixed_buffer_size is None:
        dist.broadcast(buffer_length, src=src, group=process_group)

        if rank != src:
            size = int(buffer_length.item())
            buffer = torch.empty((size), dtype=torch.uint8, device=device)

    elif rank != src:
        buffer = torch.empty((fixed_buffer_size), dtype=torch.uint8, device=device)

    dist.broadcast(buffer, src=src, group=process_group)
    buffer_list = buffer.tolist()
    null_index = next(
        (i for i, x in enumerate(buffer_list) if x == 0), len(buffer_list)
    )
    if null_index == 0:
        truncated_buffer = buffer_list
    else:
        truncated_buffer = buffer_list[:null_index]
    string = bytes(truncated_buffer).decode(encoding="utf-8", errors="strict")
    return string


def all_gather_str(
    val: str, process_group: Optional[dist.ProcessGroup] = None
) -> List[str]:
    """
    Optimized all gather-ing string without invoking all_gather_object
    which is subject to hang issues on nccl.

    Args:
        val: string to include in all_gather
        process_group: the process group to broadcast in

    Returns:
        List of all strings

    Note:
        Will construct and use a temporary gloo process group to minimize device to host transfers

    TODO: support fixed_buffer_size
    """

    if not dist.is_available() or not dist.is_initialized():
        return [val]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # use gloo so that we avoid gpu->cpu (device to host) transfers
    # with get_or_create_gloo_pg(process_group) as gloo_pg:

    # Initialize buffer and buffer_length for all ranks
    buffer = torch.frombuffer(val.encode("utf-8"), dtype=torch.uint8).to(device)
    # use `all_gather_tensors` which handles all gathering tensors
    # of same shape but different lengths (since strings may be different
    # length on each rank)
    buffer_strings = all_gather_tensors(buffer, group=process_group)

    result = [bytes(buffer.tolist()).decode("utf-8") for buffer in buffer_strings]

    return result


@contextmanager
def get_or_create_gloo_pg(
    candidate_pg: Optional[dist.ProcessGroup] = None,
) -> Generator[Optional[dist.ProcessGroup], None, None]:
    """
    Context manager to ensure that a gloo process group is used for the contained operations. First checks if the
    WORLD process group, or the provided candidate process group, is already gloo-based. In case it is, that is returned.
    Otherwise, a new gloo process group will be created and returned. Upon exiting the context, if a new process group
    was created, it will be destroyed.

    Note: If the distributed environment is not initialized, this context manager will return None and will be no-op.

    Args:
        candidate_pg: Optional process group to check if it is gloo-based. If None, the WORLD process group will be checked.
    """
    gloo_pg_created = False

    if not dist.is_initialized():
        logger.info("Not in a distributed environment, gloo process group not created")
        pg = None

    else:
        pg = candidate_pg or dist.group.WORLD
        if dist.get_backend(pg) != dist.Backend.GLOO:
            logger.info("Creating temporary gloo process group")
            pg = dist.new_group(
                timeout=timedelta(seconds=3600), backend=dist.Backend.GLOO
            )
            gloo_pg_created = True

    try:
        yield pg

    finally:
        # Cleanup temporary gloo pg if it was created
        if gloo_pg_created:
            dist.destroy_process_group(pg)
            logger.info("Destroyed temporary gloo process group")
