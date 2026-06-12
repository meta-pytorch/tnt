#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
import pickle
from typing import Any, Callable, Dict, Optional, Union

import torch
from torchtnt.utils.distributed import get_global_rank
from torchtnt.utils.fsspec import get_filesystem

logger: logging.Logger = logging.getLogger(__name__)


def is_out_of_cpu_memory(exception: BaseException) -> bool:
    """Returns True if the exception is related to CPU OOM"""
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )


def is_out_of_cuda_memory(exception: BaseException) -> bool:
    """Returns True if the exception is related to CUDA OOM"""
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and (
            "RuntimeError: cuda runtime error (2) : out of memory" in exception.args[0]
            or "CUDA out of memory." in exception.args[0]
        )
    )


def is_out_of_memory_error(exception: BaseException) -> bool:
    """Returns True if an exception is due to an OOM based on error message"""
    return is_out_of_cpu_memory(exception) or is_out_of_cuda_memory(exception)


def _bytes_to_mb_gb(num_bytes: int) -> str:
    if num_bytes < 1024 * 1024:
        return f"{round(num_bytes / (1024 * 1024), 2)} MB"
    else:
        return f"{round(num_bytes / (1024 * 1024 * 1024), 2)} GB"


def _oom_observer(
    output_dir: str,
) -> Callable[[Union[int, torch.device], int, int, int], None]:
    def oom_logger(
        device: Union[int, torch.device],
        alloc: int,
        device_alloc: int,
        device_free: int,
    ) -> None:
        """
        Log memory snapshot in the event of CUDA OOM.
        """
        logger.info(
            f"Saving memory snapshot device: {device}, alloc: {_bytes_to_mb_gb(alloc)}, device_alloc: {_bytes_to_mb_gb(device_alloc)}, device_free: {_bytes_to_mb_gb(device_free)}"
        )
        try:
            log_memory_snapshot(output_dir, "oom")
        except Exception as e:
            logger.error(f"Failed to log memory snapshot during OOM {e}")

    return oom_logger


def log_memory_snapshot(output_dir: str, file_prefix: Optional[str] = None) -> None:
    """Writes the memory snapshots to the provided ``output_dir``.
    For more information, see this `blog post <https://zdevito.github.io/2022/08/16/memory-snapshots.html>`_ .

    Args:
        output_dir: The directory to save the memory snapshot.
        file_prefix: Prefix to append to the filename. If not set, defaults to "memory_snapshot".

    Note:
        Outputs are only saved if running on a host with CUDA devices available.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not logging snapshot")
        return

    rank = get_global_rank()
    if file_prefix is None:
        file_prefix = "memory_snapshot"
    save_dir = os.path.join(output_dir, f"{file_prefix}_rank{rank}")
    try:
        snapshot = torch.cuda.memory._snapshot()
        # pyrefly: ignore [bad-argument-type]
        _dump_snapshot(save_dir, snapshot)
        logger.info(f"Logged memory snapshot to {save_dir}")
    except Exception as e:
        logger.error(f"Failed to log memory snapshot to {save_dir}: {e}")


def _dump_snapshot(save_dir: str, snapshot: Dict[str, Any]) -> None:
    fs = get_filesystem(save_dir)
    fs.mkdirs(save_dir, exist_ok=True)
    with fs.open(os.path.join(save_dir, "snapshot.pickle"), "wb") as f:
        pickle.dump(snapshot, f)
    with fs.open(os.path.join(save_dir, "trace_plot.html"), "w") as f:
        f.write(torch.cuda._memory_viz.trace_plot(snapshot))
    with fs.open(os.path.join(save_dir, "segment_plot.html"), "w") as f:
        f.write(torch.cuda._memory_viz.segment_plot(snapshot))


def attach_oom_observer(output_dir: str, trace_max_entries: int = 1000000) -> None:
    """Attaches a function to record the PyTorch memory snapshot when an out of memory error occurs.

    For more information, see this `blog post <https://zdevito.github.io/2022/08/16/memory-snapshots.html>`_ .

    Args:
        output_dir (str): The directory to save the memory snapshot.
        trace_max_entries (int, optional): The maximum number of trace entries to record. Defaults to 1000000.

    Note:
        Outputs are only saved if running on a host with CUDA devices available.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not attaching OOM observer.")
        return

    torch.cuda.memory._record_memory_history(
        enabled="all", max_entries=trace_max_entries
    )
    # pyre-ignore[16]: Module `_C` has no attribute
    #  `_cuda_attach_out_of_memory_observer`.
    torch._C._cuda_attach_out_of_memory_observer(_oom_observer(output_dir))
