# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import TracebackType
from typing import Optional, Type

import torch
from torchtnt.utils.oom import attach_oom_observer, log_memory_snapshot

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshotParams:
    """
    Memory snapshot parameters.

    Args:
        start_step: Step from which to start recording memory history.
        stop_step: Step after which to dump memory snapshot, and stop recording memory history.
        max_entries: Maximum number of events to keep in memory.
        enable_oom_observer: Whether to attach an observer to record OOM events. If stop_step is set, the
            OOM observer will only be active until stop_step is reached.

    Note: If you set enable_oom_observer to True, you don't necessarily have to set a start_step as attach_oom_observer
        will start recording memory history. Note that if you don't set a stop_step, it will continue recording memory
        history until the program exits, which may incur a slight performance cost.
    """

    start_step: Optional[int] = None
    stop_step: Optional[int] = 2
    max_entries: int = 100000
    enable_oom_observer: bool = True


class MemorySnapshotProfilerBase(ABC):
    """
    Base class for memory snapshot profiler.
    """

    def __enter__(self) -> None:
        self.start()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> Optional[bool]:
        self.stop()

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def step(self) -> None: ...


class MemorySnapshotProfiler(MemorySnapshotProfilerBase):
    """
    Records a history of memory allocation and free events, and dumps to a
    file which can be visualized offline. It by default keeps track of
    100000 events, and dumps at user specified step as well as on OOM.
    The profiler is ideally started before model is instantiated in memory
    so that allocations that stay constant in training are accounted for.

    Args:
        output_dir: Directory where to save the memory snapshots.
        memory_snapshot_params: Instance of MemorySnapshotParams.

    Raises:
        ValueError: If `start_step` is negative, or `stop_step` is less than or equal to zero.
        ValueError: If `start_step` is greater than or equal to `stop_step`.
        ValueError: If `start_step` is set and `stop_step` is not set.
        ValueError: If `stop_step` is set and neither `start_step` nor `enable_oom_observer` are set.
        ValueError: If `enable_oom_observer` is False and neither `start_step` nor `stop_step` is set

    Examples::
        memory_snapshot_params = MemorySnapshotParams(start_step=5, stop_step=10, enable_oom_observer=True)
        memory_snapshot_profiler = MemorySnapshotProfiler(output_dir="/tmp", memory_snapshot_params=memory_snapshot_params)
        for batch in dataloader:
            ...
            memory_snapshot_profiler.step()
    """

    def __init__(
        self,
        output_dir: str,
        memory_snapshot_params: Optional[MemorySnapshotParams] = None,
    ) -> None:
        super().__init__()
        self.output_dir: str = output_dir
        self.params: MemorySnapshotParams = (
            memory_snapshot_params or MemorySnapshotParams()
        )
        start_step = self.params.start_step
        stop_step = self.params.stop_step
        if start_step is not None:
            if start_step < 0:
                raise ValueError("start_step must be nonnegative.")
            elif stop_step is None:
                raise ValueError("stop_step must be specified when start_step is set.")
            elif start_step >= stop_step:
                raise ValueError("start_step must be < stop_step.")
        if stop_step is not None:
            if stop_step <= 0:
                raise ValueError("stop_step must be positive.")
            elif start_step is None and not self.params.enable_oom_observer:
                raise ValueError(
                    "stop_step must be enabled with either start_step or enable_oom_observer."
                )
        if (
            start_step is None
            and stop_step is None
            and not self.params.enable_oom_observer
        ):
            raise ValueError(
                "At least one of start_step/stop_step or enable_oom_observer must be set."
            )

        self.step_num: int = 0
        self.is_started: bool = False

        if self.params.enable_oom_observer:
            attach_oom_observer(
                output_dir=output_dir, trace_max_entries=self.params.max_entries
            )
        if start_step is not None and start_step == 0:
            self.start()

        logger.info(
            f"Created MemorySnapshotProfiler with MemorySnapshotParams={self.params}."
        )

    def start(self) -> None:
        if self.is_started:
            return
        if not torch.cuda.is_available():
            logger.warning("CUDA unavailable. Not recording memory history.")
            return

        logger.info("Starting to record memory history.")
        torch.cuda.memory._record_memory_history(max_entries=self.params.max_entries)
        self.is_started = True

    def stop(self) -> None:
        if not self.is_started:
            return
        if not torch.cuda.is_available():
            logger.warning("CUDA unavailable. Not recording memory history.")
            return

        logger.info("Stopping recording memory history.")
        torch.cuda.memory._record_memory_history(enabled=None)
        self.is_started = False

    def step(self) -> None:
        self.step_num += 1
        if (
            self.params.start_step is not None
            and self.step_num == self.params.start_step
        ):
            self.start()
        if self.params.stop_step is not None and self.step_num == self.params.stop_step:
            self.log_memory_snapshot()
            self.stop()

    def log_memory_snapshot(self) -> None:
        log_memory_snapshot(
            output_dir=self.output_dir, file_prefix=f"step_{self.step_num}"
        )
