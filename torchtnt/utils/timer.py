#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import datetime
import logging
import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import total_ordering
from time import perf_counter
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
)

import numpy as np
import torch
import torch.distributed as dist
from tabulate import tabulate
from torch.distributed.distributed_c10d import Work
from torchtnt.utils.distributed import PGWrapper


logger: logging.Logger = logging.getLogger(__name__)


@contextmanager
def log_elapsed_time(
    action_name: str, *, cuda_sync: Optional[bool] = None
) -> Generator[None, None, None]:
    """Utility to measure and log elapsed time for a given event.

    Args:
        action_name: the name of the event being timed.
        cuda_sync: Whether to synchronize the stream in order to measure the most accurate timings on CUDA. Defaults to True if CUDA is available.

    Raises:
        ValueError: If cuda_sync is set to True but CUDA is not available.
    """
    if cuda_sync and not torch.cuda.is_available():
        raise ValueError(
            "CUDA must be available in order to enable CUDA synchronization."
        )
    cuda_sync = cuda_sync if cuda_sync is not None else torch.cuda.is_available()
    try:
        if cuda_sync:
            torch.cuda.synchronize()
        start_time: float = perf_counter()
        yield
    finally:
        if cuda_sync:
            torch.cuda.synchronize()
        interval_time: float = perf_counter() - start_time
        logger.info(f"{action_name} took {interval_time} seconds")


@total_ordering
@dataclass
class TimedActionStats:
    """Dataclass for storing timed action stats. These can be consumed by report generation methods, so metrics should be aggregated."""

    action_name: str
    mean_duration: float = 0.0
    num_calls: int = 0
    total_duration: float = 0.0
    percentage_of_total_time: float = 0.0

    def __le__(self, other: "TimedActionStats") -> bool:
        return self.percentage_of_total_time <= other.percentage_of_total_time


@dataclass
class TimerReport:
    timed_action_stats: List[TimedActionStats]
    total_calls: int
    total_duration: float


@runtime_checkable
class TimerProtocol(Protocol):
    """
    Defines a Timer Protocol with `time` and `reset` methods and an attribute `recorded_durations` for storing timings.
    """

    recorded_durations: Dict[str, List[float]]

    @contextmanager
    def time(self, action_name: str) -> Generator[None, None, None]:
        """
        A context manager for timing a code block.

        Args:
            action_name: the name under which to store the timing of what is enclosed in the context manager.
        """
        ...

    def reset(self) -> None:
        """
        A method to reset the state of the Timer.
        """
        ...

    def _make_report(self) -> TimerReport:
        """
        Creates a report of timing data.
        """
        ...


class Timer(TimerProtocol):
    def __init__(
        self,
        *,
        cuda_sync: Optional[bool] = None,
        verbose: bool = False,
    ) -> None:
        """
        A Timer class which implements TimerProtocol and stores timings in a dictionary `recorded_durations`.

        Args:
            cuda_sync: whether to call torch.cuda.synchronize() before and after timing. Defaults to True if CUDA is available.
            verbose: whether to enable verbose logging.

        Note:
            Enabling cuda_sync will incur a performance hit, but will ensure accurate timings on GPUs.

        Raises:
            ValueError: If cuda_sync is set to True but CUDA is not available.

        """
        if cuda_sync and not torch.cuda.is_available():
            raise ValueError(
                "CUDA must be available in order to enable CUDA synchronization."
            )
        self.cuda_sync: bool = cuda_sync if cuda_sync is not None else False
        self.verbose = verbose
        self.recorded_durations: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def time(
        self,
        action_name: str,
    ) -> Generator[None, None, None]:
        """
        A context manager for timing a code block, with optional cuda synchronization and verbose timing.

        Args:
            action_name: the name under which to store the timing of what is enclosed in the context manager.
        """
        start_time: float = perf_counter()
        try:
            if self.cuda_sync:
                torch.cuda.synchronize()
            yield
        finally:
            if self.cuda_sync:
                torch.cuda.synchronize()
            interval_time: float = perf_counter() - start_time
            if self.verbose:
                logger.info(f"{action_name} took {interval_time} seconds.")
        self.recorded_durations[action_name].append(interval_time)

    def reset(self) -> None:
        """
        Reset the recorded_durations to an empty list
        """
        self.recorded_durations = defaultdict(list)

    def _make_report(self: TimerProtocol) -> TimerReport:
        total_time = 0.0
        for _, durations in self.recorded_durations.items():
            array_value = np.array(durations)
            array_sum = np.sum(array_value)
            total_time += array_sum

        action_stats = [
            TimedActionStats(
                action_name=a,
                mean_duration=np.mean(d),
                num_calls=len(d),
                total_duration=np.sum(d),
                percentage_of_total_time=100.0 * np.sum(d) / total_time,
            )
            for a, d in self.recorded_durations.items()
        ]
        action_stats.sort(reverse=True)
        total_calls = sum(x.num_calls for x in action_stats)
        return TimerReport(
            timed_action_stats=action_stats,
            total_calls=total_calls,
            total_duration=total_time,
        )


class AggregatedTimer(Timer):
    """
    A Timer class which implements TimerProtocol and stores aggregated timing stats. Instead of
    storing the recorded durations for each action, it accumulates running metrics and computes
    final stats at the end when generating the report. This is useful for cases where the number
    of samples is too large to store in memory.
    """

    def __init__(
        self,
        cuda_sync: Optional[bool] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(cuda_sync=cuda_sync, verbose=verbose)
        self._aggregate_stats: Dict[str, TimedActionStats] = defaultdict(
            lambda: TimedActionStats(action_name="")  # Filled in on time() method
        )

    @contextmanager
    def time(
        self,
        action_name: str,
    ) -> Generator[None, None, None]:
        # Run base class context manager first
        with super().time(action_name):
            yield

        # Update aggregate stats
        latest_duration: float = self.recorded_durations[action_name][-1]
        self._aggregate_stats[action_name].action_name = action_name
        self._aggregate_stats[action_name].num_calls += 1
        self._aggregate_stats[action_name].total_duration += latest_duration

        # Reset recorded durations to avoid storing data
        self.recorded_durations.clear()

    def _make_report(self) -> TimerReport:
        """
        Creates the report but considering that the data is aggregated in the correct structure.
        """
        total_time = 0.0
        total_calls = 0

        # Calculate total time and calls across all actions
        for stats in self._aggregate_stats.values():
            total_time += stats.total_duration
            total_calls += stats.num_calls

        # Build report data
        action_stats: List[TimedActionStats] = []
        for stats in self._aggregate_stats.values():
            stats.mean_duration = (
                stats.total_duration / stats.num_calls if stats.num_calls > 0 else 0.0
            )
            stats.percentage_of_total_time = (
                100.0 * stats.total_duration / total_time if total_time > 0 else 0.0
            )
            action_stats.append(stats)

        # Sort by percentage (descending)
        action_stats.sort(key=lambda x: x.percentage_of_total_time, reverse=True)

        return TimerReport(
            timed_action_stats=action_stats,
            total_calls=total_calls,
            total_duration=total_time,
        )


class BoundedTimer(Timer):
    """
    A Timer class which implements TimerProtocol and stores timings in a dictionary `recorded_durations`.

    Same behavior as timer, but with the addition of size_bounds = (lower, upper)

    Args:
        ...
        size_bounds: defines the range of samples that should be kept in the timer. The lower bound should be smaller than
            the upper bound. When the number of samples reaches the upper bound, the oldest (upper-lower) bound samples will
            be removed. This range is applied per action.
    """

    def __init__(self, lower_bound: int, upper_bound: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert lower_bound > 0
        assert lower_bound < upper_bound
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @contextmanager
    def time(
        self,
        action_name: str,
    ) -> Generator[None, None, None]:
        with super().time(action_name):
            yield
        self._apply_bounds(action_name)

    def _apply_bounds(self, action_name: str) -> None:
        # Keep 'lower_bound' most recent samples, if at or over upper bound
        n_samples: int = len(self.recorded_durations[action_name])
        if self.upper_bound <= n_samples:
            self.recorded_durations[action_name] = list(
                self.recorded_durations[action_name][-self.lower_bound :]
            )


def get_timer_summary(timer: TimerProtocol) -> str:
    """Given a timer, generate a summary of all the recorded actions.

    Args:
        timer: the Timer object for which to generate a summary

    Raises:
        ValueError
            If the input Timer has no recorded actions
    """
    report: TimerReport = timer._make_report()

    sep: str = os.linesep
    output_string = f"Timer Report{sep}"

    # Handle empty timer case
    if not report.timed_action_stats:
        return output_string

    max_key = max(len(a.action_name) for a in report.timed_action_stats)

    # pyre-fixme[53]: Captured variable `max_key` is not annotated.
    def log_row(action: str, mean: str, num_calls: str, total: str, per: str) -> str:
        row = f"{sep}|  {action:<{max_key}s}\t|  {mean:<15}\t|"
        row += f"  {num_calls:<15}\t|  {total:<15}\t|  {per:<15}\t|"
        return row

    header_string = log_row(
        "Action",
        "Mean duration (s)",
        "Num calls",
        "Total time (s)",
        "Percentage %",
    )

    output_string_len = len(header_string.expandtabs()) - 1
    sep_lines = f"{sep}{'-' * output_string_len}"
    output_string += sep_lines + header_string + sep_lines
    output_string += log_row(
        "Total", "-", f"{report.total_calls:}", f"{report.total_duration:.5}", "100 %"
    )
    output_string += sep_lines

    for action in report.timed_action_stats:
        output_string += log_row(
            action.action_name,
            f"{action.mean_duration:.5}",
            f"{action.num_calls}",
            f"{action.total_duration:.5}",
            f"{action.percentage_of_total_time:.5}",
        )
    output_string += sep_lines

    output_string += sep
    return output_string


def get_durations_histogram(
    recorded_durations: Dict[str, List[float]],
    percentiles: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    """Computes a histogram of percentiles from the recorded durations passed in.

    Args:
        recorded_durations: The mapping of durations to sync and compute histograms from.
        percentiles: The percentiles to compute. Values should be in the range [0, 100].

    Returns:
        A dictionary mapping the action names to a dictionary of the computed percentiles, along with the mean duration of each action.

    Raises:
        ValueError: If the input percentiles are not in the range [0, 100].
    """
    _validate_percentiles(percentiles)
    percentiles = sorted(percentiles)
    return _compute_percentiles(recorded_durations, percentiles=percentiles)


def get_synced_durations_histogram(
    recorded_durations: Dict[str, List[float]],
    percentiles: Sequence[float],
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Dict[str, float]]:
    """Synchronizes the recorded durations across ranks.

    Args:
        recorded_durations: The mapping of durations to sync and compute histograms from.
        percentiles: The percentiles to compute. Values should be in the range [0, 100].
        pg (optional): The process group to use for synchronization. Defaults to the global process group.

    Returns:
        A dictionary mapping the action names to a dictionary of the computed percentiles, along with the mean duration of each action.

    Raises:
        ValueError: If the input percentiles are not in the range [0, 100].
    """
    _validate_percentiles(percentiles)
    synced_durations = _sync_durations(recorded_durations, pg)
    return get_durations_histogram(synced_durations, percentiles=percentiles)


def get_synced_timer_histogram(
    timer: TimerProtocol,
    percentiles: Sequence[float],
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Dict[str, float]]:
    """Synchronizes the input timer's recorded durations across ranks.

    Args:
        timer: The TimerProtocol object whose recorded durations will be synced.
        percentiles: The percentiles to compute. Values should be in the range [0, 100].
        pg (optional): The process group to use for synchronization. Defaults to the global process group.

    Returns:
        A dictionary mapping the action names to a dictionary of the computed percentiles, along with the mean duration of each action.

    Raises:
        ValueError: If the input percentiles are not in the range [0, 100].
    """
    return get_synced_durations_histogram(
        timer.recorded_durations, percentiles=percentiles, pg=pg
    )


def _sync_durations(
    recorded_durations: Dict[str, List[float]], pg: Optional[dist.ProcessGroup]
) -> Dict[str, List[float]]:
    if not (dist.is_available() and dist.is_initialized()):
        return recorded_durations

    pg_wrapper = PGWrapper(pg)
    world_size = pg_wrapper.get_world_size()
    outputs = [None] * world_size
    pg_wrapper.all_gather_object(outputs, recorded_durations)
    ret = defaultdict(list)
    for output in outputs:
        if not output:
            continue
        for k, v in output.items():
            if k not in ret:
                ret[k] = []
            ret[k].extend(v)
    return ret


def _compute_percentiles(
    durations: Dict[str, List[float]], percentiles: Sequence[float]
) -> Dict[str, Dict[str, float]]:
    ret = {}
    for name, values in durations.items():
        ret[name] = _compute_percentile(name, values, percentiles=percentiles)
    return ret


def _compute_percentile(
    name: str, timings: List[float], percentiles: Sequence[float]
) -> Dict[str, float]:
    ret = {}

    # By default, numpy's percentile function will interpolate between values,
    # but we want to snap to actual metrics that were recorded. For more
    # discussion of percentile interpolation, see:
    # https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
    computed_percentiles = np.percentile(timings, percentiles, interpolation="lower")

    # computed_percentiles is a sequence of floats with the percentile
    # results. We use enumerate to allow us to grab the index for each
    # computed percentile, so that we can grab the corresponding percentile
    # to use as the "name" when we turn these values into the metrics.
    for i, percentile_value in enumerate(computed_percentiles):
        percentile = percentiles[i]

        ret[f"p{percentile}"] = percentile_value

    # include the mean as well in addition to the percentiles passed in
    ret["avg"] = np.mean(timings)
    return ret


def _validate_percentiles(percentiles: Sequence[float]) -> None:
    for p in percentiles:
        if p < 0 or p > 100:
            raise ValueError(f"Percentile must be between 0 and 100. Got {p}")


def get_recorded_durations_table(result: Dict[str, Dict[str, float]]) -> str:
    r"""
    Helper function to generate recorded duration time in tabular format
    """
    if len(result) == 0:
        return ""
    sub_dict = next(iter(result.values()))
    if len(sub_dict) == 0:
        return ""
    column_headers = ["Name"] + list(sub_dict.keys())
    row_output = []
    for key in result:
        row = [key] + ["{:.3f}".format(x) for x in result[key].values()]
        row_output.append(row)
    tabulate_output = tabulate(
        row_output,
        tablefmt="pipe",
        headers=column_headers,
    )
    return "\n" + tabulate_output


class FullSyncPeriodicTimer:
    """
    Measures time (resets if given interval elapses) on rank 0
    and propagates result to other ranks.
    Propagation is done asynchronously from previous step
    in order to avoid blocking of a training process.
    """

    def __init__(self, interval: datetime.timedelta, cpu_pg: dist.ProcessGroup) -> None:
        self._interval = interval
        self._cpu_pg = cpu_pg
        self._prev_time: float = perf_counter()
        self._timeout_tensor: torch.Tensor = torch.zeros(1, dtype=torch.int)
        self._prev_work: Optional[Work] = None

    def check(self) -> bool:
        ret = False
        curr_time = perf_counter()

        if self._prev_work is not None:
            self._prev_work.wait()
            ret = self._timeout_tensor[0].item() == 1
            if ret:
                self._prev_time = curr_time

        self._timeout_tensor[0] = (
            1 if (curr_time - self._prev_time) >= self._interval.total_seconds() else 0
        )
        self._prev_work = dist.broadcast(
            self._timeout_tensor, 0, group=self._cpu_pg, async_op=True
        )

        return ret

    def wait_remaining_work(self) -> None:
        if self._prev_work is not None:
            self._prev_work.wait()
