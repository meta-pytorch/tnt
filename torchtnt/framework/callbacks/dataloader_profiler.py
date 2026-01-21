# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import time
from typing import Optional

from pyre_extensions import none_throws

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TTrainUnit
from torchtnt.utils.distributed import get_global_rank
from torchtnt.utils.loggers.logger import MetricLogger

logger: logging.Logger = logging.getLogger(__name__)


def _empty_train_step(state: State, data: object) -> None:
    return


class DataloaderProfiler(Callback):
    """
    This callback is used to profile the dataloader together with the trainer.
    It reports the QPS, train iteration time and time wait for batch.
    It is useful to understand how much slowdown might be caused by the dataloader, and allow to tune the batch size and other related params accordingly.
    Note that it's not meant to be run side by side with training, and will terminate the loop once it's done.

    Args:
        mode: The mode to run the profiler in. Possible values:
            DATALOADER_ONLY - profile only the dataloader by avoiding running the train step.
        num_steps_to_profile: The number of steps to profile. The loop will be terminated after this many steps.
        batch_size: The batch size used by the dataloader.
        logger: a MetricLogger to log the metrics (QPS, iteration time, time wait for batch) to.
        logging_prefix: an optional prefix to add to the logged metrics names.
        should_report_time_wait_for_batch: whether to report the time wait for batch. Default is True.
        should_report_iteration_time: whether to report the iteration time. Default is True.
    """

    DATALOADER_ONLY = "DATALOADER_ONLY"

    def __init__(
        self,
        *,
        mode: str,
        num_steps_to_profile: int,
        batch_size: int,
        logger: MetricLogger,
        logging_prefix: Optional[str] = "DL Profiler:",
        should_report_time_wait_for_batch: Optional[bool] = True,
        should_report_iteration_time: Optional[bool] = True,
    ) -> None:
        if mode not in {
            self.DATALOADER_ONLY,
        }:
            raise ValueError(f"Invalide mode: {mode}")

        self._mode = mode

        if num_steps_to_profile <= 0:
            raise ValueError(
                f"Expected num_steps_to_profile to be a positive integer, but got {num_steps_to_profile}"
            )

        if batch_size <= 0:
            raise ValueError(
                f"Expected batch_size to be a positive integer, but got {batch_size}"
            )

        self._num_steps_to_profile: int = num_steps_to_profile
        self._batch_size = batch_size
        self._logger = logger
        self._logging_prefix = logging_prefix
        self._should_report_twfb = should_report_time_wait_for_batch
        self._should_report_iteration_time = should_report_iteration_time

        self._last_time: float = 0.0
        self._rank: int = get_global_rank()

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        if self._mode == self.DATALOADER_ONLY:
            unit.train_step = _empty_train_step  # pyre-ignore. Overriding the train step in order to ensure we're only profiling the dataloader

        self._last_time = time.perf_counter()

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        number_of_steps_completed = unit.train_progress.num_steps_completed

        if number_of_steps_completed >= self._num_steps_to_profile:
            logger.info("Completed profiling. Indicating loop to stop.")
            state.stop()

        self._report_qps(number_of_steps_completed)
        if self._should_report_twfb:
            self._report_twfb(state, number_of_steps_completed)

    def on_train_step_start(self, state: State, unit: TTrainUnit) -> None:
        self._maybe_report_iteration_time(state, unit)

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        self._maybe_report_iteration_time(state, unit)

    def _maybe_report_iteration_time(self, state: State, unit: TTrainUnit) -> None:
        if not self._should_report_iteration_time:
            return

        number_of_steps_completed = unit.train_progress.num_steps_completed
        if number_of_steps_completed == 0:
            return

        self._report_iteration_time(state, number_of_steps_completed)

    def _report_iteration_time(self, state: State, step_logging_for: int) -> None:
        timer = none_throws(state.train_state).iteration_timer
        time_list = timer.recorded_durations.get("train_iteration_time", [])
        if not time_list:
            logger.warning(
                "Unexpected falsy time list. Skipping iteration time logging"
            )
            return

        if self._rank == 0:
            self._logger.log(
                f"{self._logging_prefix} Train iteration time (seconds)",
                time_list[-1],
                step_logging_for,
            )

    def _report_twfb(self, state: State, step: int) -> None:
        timer = none_throws(state.train_state).iteration_timer
        time_list = timer.recorded_durations.get("data_wait_time", [])
        if not time_list:
            logger.warning(
                "Unexpected falsy time list. Skipping time wait for batch logging"
            )
            return

        if self._rank == 0:
            self._logger.log(
                f"{self._logging_prefix} Time waiting for batch (seconds)",
                time_list[-1],
                step,
            )

    def _report_qps(self, step: int) -> None:
        curr_time = time.perf_counter()
        time_elapsed = curr_time - self._last_time
        qps = self._batch_size / time_elapsed

        if self._rank == 0:
            self._logger.log(f"{self._logging_prefix} QPS", qps, step)

        self._last_time = curr_time
