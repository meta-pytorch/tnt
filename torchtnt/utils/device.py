#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
import shutil
import subprocess
from collections import defaultdict
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Mapping, Optional, TypeVar

import torch
from typing_extensions import Protocol, runtime_checkable, TypedDict

logger: logging.Logger = logging.getLogger(__name__)


def get_device_from_env() -> torch.device:
    """Function that gets the torch.device based on the current environment.

    This currently supports only CPU, GPU, and MPS devices. If CUDA is available, this function also sets the CUDA device.

    Within a distributed context, this function relies on the ``LOCAL_RANK`` environment variable
    to be made available by the program launcher for setting the appropriate device index.

    Raises:
        RuntimeError
            If ``LOCAL_RANK`` is outside the range of available GPU devices.
    """
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                "The local rank is larger than the number of available GPUs."
            )
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


T = TypeVar("T")
TSelf = TypeVar("TSelf")


def _is_named_tuple(x: T) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields")


def copy_data_to_device(
    data: T,
    device: torch.device,
    stream_to_record: Optional[torch.cuda.Stream] = None,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Function that recursively copies data to a torch.device.

    Args:
        data: The data to copy to device
        device: The device to which the data should be copied
        stream_to_record: The CUDA stream to which the data should be recorded. Useful if this function is called
            on side stream, and the data is expected to be used on the main stream.
        args: positional arguments that will be passed to the `to` call
        kwargs: keyword arguments that will be passed to the `to` call

    Returns:
        The data on the correct device
    """

    data_type = type(data)
    if issubclass(data_type, defaultdict):
        return data_type(
            data.default_factory,
            {
                k: copy_data_to_device(v, device, *args, **kwargs)
                for k, v in data.items()
            },
        )
    elif (
        hasattr(data, "items")
        and hasattr(data, "__getitem__")
        and hasattr(data, "__iter__")
    ):
        # pyre-ignore: Too many arguments [19]: Call `object.__init__` expects 0 positional arguments, 1
        return data_type(
            {
                k: copy_data_to_device(v, device, *args, **kwargs)
                # pyre-ignore: Undefined attribute [16]: `Variable[T]` has no attribute `items`.
                for k, v in data.items()
            }
        )
    elif issubclass(data_type, list):
        return data_type(copy_data_to_device(e, device, *args, **kwargs) for e in data)
    elif issubclass(data_type, tuple):
        if hasattr(data, "_asdict") and hasattr(data, "_fields"):
            return data_type(
                **copy_data_to_device(data._asdict(), device, *args, **kwargs)
            )
        return data_type(copy_data_to_device(e, device, *args, **kwargs) for e in data)
    # checking for __dataclass_fields__ is official way to check if data is a dataclass
    elif hasattr(data, "__dataclass_fields__"):
        new_data_class = data_type(
            **{
                field.name: copy_data_to_device(
                    getattr(data, field.name), device, *args, **kwargs
                )
                for field in fields(data)
                if field.init
            }
        )
        for field in fields(data):
            if not field.init:
                setattr(
                    new_data_class,
                    field.name,
                    copy_data_to_device(
                        getattr(data, field.name), device, *args, **kwargs
                    ),
                )
        return new_data_class
    elif hasattr(data, "to"):
        # pyre-ignore Undefined attribute [16]: `Variable[T]` has no attribute `to`
        gpu_data = data.to(device, *args, **kwargs)
        if stream_to_record is not None and hasattr(gpu_data, "record_stream"):
            gpu_data.record_stream(stream_to_record)
        return gpu_data

    return data


def record_data_in_stream(data: T, stream: torch.cuda.streams.Stream) -> None:
    """
    Records the tensor element on certain streams, to avoid memory from being reused for another tensor.
    As mentioned in
    https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html, PyTorch
    uses the "caching allocator" for memory allocation for tensors. When a tensor is
    freed, its memory is likely to be reused by newly constructed tensors. By default,
    this allocator traces whether a tensor is still in use by only the CUDA stream where
    it was created. When a tensor is used by additional CUDA streams, we need to call
    `record_stream` to tell the allocator about these streams. Otherwise, the allocator
    might free the underlying memory of the tensor once it is no longer used by the
    creator stream. This is a notable programming trick when we write programs using
    multiple CUDA streams.

    Args:
        data: The data on which to call record_stream
        stream: The CUDA stream with which to call record_stream
    """

    # Redundant isinstance(data, tuple) check is required here to make pyre happy
    if _is_named_tuple(data) and isinstance(data, tuple):
        record_data_in_stream(data._asdict(), stream)
    elif isinstance(data, (list, tuple)):
        for e in data:
            record_data_in_stream(e, stream)
    elif isinstance(data, Mapping):
        for _, v in data.items():
            record_data_in_stream(v, stream)
    elif is_dataclass(data) and not isinstance(data, type):
        for field in fields(data):
            record_data_in_stream(getattr(data, field.name), stream)
    elif isinstance(data, _MultistreamableData):
        data.record_stream(stream)


@runtime_checkable
class _MultistreamableData(Protocol):
    """
    Objects implementing this interface are allowed to be transferred
    from one CUDA stream to another.
    torch.Tensor implements this interface.
    """

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        """
        See https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html
        """
        ...


class GPUStats(TypedDict):
    utilization_gpu_percent: float
    utilization_memory_percent: float
    fan_speed_percent: float
    memory_used_mb: int
    memory_free_mb: int
    temperature_gpu_celsius: float
    temperature_memory_celsius: float


def get_nvidia_smi_gpu_stats(device: torch.device) -> GPUStats:  # pragma: no-cover
    """Get GPU stats from nvidia smi.

    Args:
         device: A GPU torch.device to get stats from.

    Returns:
        dict (str, float): a dict that maps gpu stats to their values.

        Keys:
            - 'utilization_gpu_percent'
            - 'utilization_memory_percent'
            - 'fan_speed_percent'
            - 'memory_used_mb'
            - 'memory_free_mb'
            - 'temperature_gpu_celsius'
            - 'temperature_memory_celsius'

    Raises:
        FileNotFoundError:
            If nvidia-smi command is not found.
    """
    # Check for nvidia-smi
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        raise FileNotFoundError("nvidia-smi: command not found.")

    # Prepare keys
    gpu_stat_keys = [
        "utilization_gpu_percent",
        "utilization_memory_percent",
        "fan_speed_percent",
        "memory_used_mb",
        "memory_free_mb",
        "temperature_gpu_celsius",
        "temperature_memory_celsius",
    ]

    # Format as "utilization.gpu,utilization.memory,fan.speed,etc"
    smi_query = ",".join([".".join(key.split("_")[:-1]) for key in gpu_stat_keys])

    gpu_id = torch._utils._get_device_index(device)

    # Get values from nvidia-smi
    result = subprocess.run(
        [
            nvidia_smi_path,
            f"--query-gpu={smi_query}",
            "--format=csv,nounits,noheader",
            f"--id={gpu_id}",
        ],
        encoding="utf-8",
        capture_output=True,
        check=True,
    )

    # Format output
    output = result.stdout.strip()
    stats = []
    for value in output.split(", "):
        try:
            float_val = float(value)
        except ValueError:
            float_val = 0.0
        stats.append(float_val)

    # Add units to keys and populate values
    # This is not a dict comprehension to prevent pyre warnings.
    gpu_stats: GPUStats = {
        "utilization_gpu_percent": stats[0],
        "utilization_memory_percent": stats[1],
        "fan_speed_percent": stats[2],
        "memory_used_mb": stats[3],
        "memory_free_mb": stats[4],
        "temperature_gpu_celsius": stats[5],
        "temperature_memory_celsius": stats[6],
    }
    return gpu_stats


class CPUStats(TypedDict):
    cpu_vm_percent: float
    cpu_percent: float
    cpu_swap_percent: float
    worker_cpu_time_user: float
    worker_cpu_time_system: float
    worker_rss: float


def get_psutil_cpu_stats() -> CPUStats:
    """Get CPU process stats using psutil.

    Returns:
        Dict[str, float]: a dict that maps cpu stats to their values.

        Keys:

            - 'cpu_vm_percent'
            - 'cpu_percent'
            - 'cpu_swap_percent'
            - 'worker_cpu_time_user'
            - 'worker_cpu_time_system'
            - 'worker_rss'
    """
    try:
        import psutil
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "`get_cpu_process_metrics` requires `psutil` to be installed."
            " Install it by running `pip install -U psutil`."
        )

    process = psutil.Process()
    cpu_times = process.cpu_times()

    stats: CPUStats = {
        "cpu_vm_percent": psutil.virtual_memory().percent,
        "cpu_percent": psutil.cpu_percent(),
        "cpu_swap_percent": psutil.swap_memory().percent,
        "worker_cpu_time_user": cpu_times.user,
        "worker_cpu_time_system": cpu_times.system,
        "worker_rss": float(process.memory_info().rss),
    }
    return stats


def collect_system_stats(device: torch.device) -> Dict[str, Any]:
    system_stats: Dict[str, Any] = {}
    cpu_stats = get_psutil_cpu_stats()
    system_stats.update(**cpu_stats)

    if torch.cuda.is_available():
        try:
            gpu_stats = get_nvidia_smi_gpu_stats(device)
            system_stats.update(**gpu_stats)
            system_stats.update(torch.cuda.memory_stats())
        except FileNotFoundError:
            logger.warning("Unable to find nvidia-smi. Skipping GPU stats collection.")

    return system_stats


def set_float32_precision(precision: str = "high") -> None:
    """Sets the precision of float32 matrix multiplications and convolution operations.

    For more information, see the PyTorch docs:
    - https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    - https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.allow_tf32

    Args:
        precision: The setting to determine which datatypes to use for matrix multiplication and convolution operations.
    """
    if not (torch.cuda.is_available()):  # Not relevant for non-CUDA devices
        return
    # set precision for matrix multiplications
    torch.set_float32_matmul_precision(precision)
    # set precision for convolution operations
    if precision == "highest":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
