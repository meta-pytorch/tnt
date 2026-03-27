#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import time
from collections import deque
from contextlib import contextmanager
from datetime import timedelta
from enum import Enum
from threading import Event, Thread
from typing import Dict, Generator, List, Mapping, Sequence, Tuple

import psutil
import torch

_DEFAULT_MEASURE_INTERVAL = timedelta(milliseconds=100)


def _is_named_tuple(
    x: object,
) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields")


def get_tensor_size_bytes_map(
    obj: object,
) -> Dict[torch.Tensor, int]:
    tensor_map = {}
    attributes_q = deque()
    attributes_q.append(obj)
    while attributes_q:
        attribute = attributes_q.popleft()
        if isinstance(attribute, torch.Tensor):
            tensor_map[attribute] = attribute.size().numel() * attribute.element_size()
        elif _is_named_tuple(attribute):
            attributes_q.extend(attribute._asdict().values())
        elif isinstance(attribute, Mapping):
            attributes_q.extend(attribute.values())
        elif isinstance(attribute, Sequence) and not isinstance(attribute, str):
            attributes_q.extend(attribute)
        elif hasattr(attribute, "__dict__") and not isinstance(attribute, Enum):
            attributes_q.extend(attribute.__dict__.values())
    return tensor_map


class RSSProfiler:
    """A profiler that periodically measures RSS (resident set size) delta.

    The baseline RSS is measured when the profiler is initialized.
    The RSS result is stored in the rss_deltas_bytes dict of the class.

    Attributes:
        interval: The interval for measuring RSS. The default value is 100ms.
        rss_deltas_bytes: The RSS delta bytes stored as dict. Key is the name for the profiling round, value is the list of RSS delta bytes captured periodically.
    """

    def __init__(self, interval: timedelta = _DEFAULT_MEASURE_INTERVAL) -> None:
        self.rss_deltas_bytes: Dict[str, List[int]] = {}
        self.interval = interval

    @contextmanager
    def profile(self, name: str) -> Generator[None, None, None]:
        """Profile the current process and store the results with a custom name as the key.

        Profile the process by starting a separate thread to capture the RSS periodically.
        The RSS result is stored in the rss_deltas_bytes dict of the class with the provided name as the key.

        Args:
            name: The name for the profiling round.
        """
        if name not in self.rss_deltas_bytes:
            self.rss_deltas_bytes[name] = []
        thread, stop_event = _get_target_thread(
            self.rss_deltas_bytes[name], self.interval
        )
        thread.start()
        try:
            yield
        finally:
            stop_event.set()
            thread.join()

    def reset(self) -> None:
        """
        Resets the stored rss_deltas_bytes dict to empty.
        """
        self.rss_deltas_bytes = {}


@contextmanager
def measure_rss_deltas(
    rss_deltas: List[int], interval: timedelta = _DEFAULT_MEASURE_INTERVAL
) -> Generator[None, None, None]:
    """
    A context manager that periodically measures RSS (resident set size) delta.

    The baseline RSS is measured when the context manager is initialized.

    Args:
        rss_deltas: The list to which the measured RSS deltas (measured in
            bytes) are appended.
        interval: The interval at which RSS deltas are measured.
    """
    thread, stop_event = _get_target_thread(rss_deltas, interval)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()


def _get_target_thread(
    rss_deltas: List[int], interval: timedelta
) -> Tuple[Thread, Event]:
    baseline_rss_bytes = psutil.Process().memory_info().rss
    stop_event = Event()
    return (
        Thread(
            target=_measure,
            args=(
                rss_deltas,
                interval,
                baseline_rss_bytes,
                stop_event,
            ),
            name="RSSMemoryMonitor",
        ),
        stop_event,
    )


def _measure(
    rss_deltas: List[int],
    interval: timedelta,
    baseline_rss_bytes: int,
    stop_event: Event,
) -> None:
    p = psutil.Process()
    while not stop_event.is_set():
        rss_deltas.append(p.memory_info().rss - baseline_rss_bytes)
        time.sleep(interval.total_seconds())
