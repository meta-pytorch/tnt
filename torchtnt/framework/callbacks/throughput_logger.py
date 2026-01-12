# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import logging
import time
from collections import defaultdict
from typing import Dict, Mapping

from pyre_extensions import none_throws
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import ActivePhase, State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit
from torchtnt.utils.loggers.logger import MetricLogger

ACTIVE_PHASE_TO_ITERATION_TIME_KEY: Mapping[ActivePhase, str] = {
    ActivePhase.TRAIN: "train_iteration_time",
    ActivePhase.EVALUATE: "eval_iteration_time",
    ActivePhase.PREDICT: "predict_iteration_time",
}

ACTIVE_PHASE_TO_LABEL_PREFIX: Mapping[ActivePhase, str] = {
    ActivePhase.TRAIN: "Train",
    ActivePhase.EVALUATE: "Eval",
    ActivePhase.PREDICT: "Predict",
}

logger: logging.Logger = logging.getLogger(__name__)


class ThroughputLogger(Callback):
    """
    A callback which logs the train/eval/predict/fit throughput. For instance, it can be used to log QPS and number of batches processed per second.
    The callback logs the throughput on a step basis and on an epoch basis.
    We measure the throughput by dividing the number of batches processed (times the number of items in batch) by the time it took to process the batch:
        On a step granularity, we do this by leveraging the already collected timers for the iteration time and data wait time.
        On an epoch granularity, we measure the time between on_train_epoch_start and on_train_epoch_end on this callback to calculate the throughput.

    Args:
        logger: A a subclass of :class:`torchtnt.utils.loggers.logger.MetricLogger`.
        throughput_per_batch: a dict mapping the item name to the number of corresponding items in the batch.
            For instace, a user can pass in {Batches: 1, Queries: 32} which will visualize two charts -
            one for Batches per second and one for Queries per second.
            As an example, if each of your batches is of type: {data: torch.Size([16, 8, 8]), labels: torch.Size([16,1])}, then you could pass {Queries: 16}.
        log_every_n_steps: an int to control the log frequency. Default is 1.
        warmup_steps: an int to control the number of warmup steps. We will start logging only after the amount of warmup steps were completed. Default is 0.

    Note:
        The values reported are only for rank 0.
        For more accurate measurement of epoch throughput, it is recommended to place this callback at the end of the callback list.
    """

    def __init__(
        self,
        logger: MetricLogger,
        throughput_per_batch: Mapping[str, int],
        *,
        log_every_n_steps: int = 1,
        warmup_steps: int = 0,
    ) -> None:
        self._logger = logger

        if not throughput_per_batch:
            raise ValueError("throughput_per_batch cannot be empty")

        for item, num_items in throughput_per_batch.items():
            if num_items < 1:
                raise ValueError(
                    f"throughput_per_batch item {item} must be at least 1, got {num_items}"
                )

        self._throughput_per_batch = throughput_per_batch

        if log_every_n_steps < 1:
            raise ValueError(
                f"log_every_n_steps must be at least 1, got {log_every_n_steps}"
            )

        self._log_every_n_steps = log_every_n_steps

        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be at least 0, got {warmup_steps}")

        self._warmup_steps = warmup_steps

        self._epoch_start_times: Dict[ActivePhase, float] = {}
        self._steps_in_epoch: Dict[ActivePhase, int] = defaultdict(int)

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        self._maybe_log_for_step(
            state,
            unit.train_progress.num_steps_completed - 1,
        )
        self._steps_in_epoch[ActivePhase.TRAIN] += 1

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        self._maybe_log_for_step(
            state,
            unit.train_progress.num_steps_completed,
            is_step_end_hook=False,
        )

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        self._maybe_log_for_step(
            state,
            unit.eval_progress.num_steps_completed - 1,
        )
        self._steps_in_epoch[ActivePhase.EVALUATE] += 1

    def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
        self._maybe_log_for_step(
            state,
            unit.eval_progress.num_steps_completed,
            is_step_end_hook=False,
        )
        self._log_for_epoch(
            state,
            epoch_logging_for=unit.eval_progress.num_epochs_completed,
        )

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        self._maybe_log_for_step(
            state,
            unit.predict_progress.num_steps_completed - 1,
        )

    def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
        self._maybe_log_for_step(
            state,
            unit.predict_progress.num_steps_completed,
            is_step_end_hook=False,
        )
        self._log_for_epoch(
            state,
            epoch_logging_for=unit.predict_progress.num_epochs_completed,
        )

    def on_train_epoch_start(self, state: State, unit: TTrainUnit) -> None:
        self._epoch_start_times[ActivePhase.TRAIN] = time.perf_counter()

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        self._log_for_epoch(
            state,
            epoch_logging_for=unit.train_progress.num_epochs_completed,
        )

    def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
        self._epoch_start_times[ActivePhase.EVALUATE] = time.perf_counter()

    def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
        self._epoch_start_times[ActivePhase.PREDICT] = time.perf_counter()

    def _maybe_log_for_step(
        self,
        state: State,
        step_logging_for: int,
        *,
        is_step_end_hook: bool = True,
    ) -> None:
        if step_logging_for <= self._warmup_steps:
            return

        if step_logging_for % self._log_every_n_steps != 0:
            return

        active_phase_state = none_throws(state.active_phase_state())
        timer_recorded_durations = active_phase_state.iteration_timer.recorded_durations
        iteration_time_list = timer_recorded_durations.get(
            ACTIVE_PHASE_TO_ITERATION_TIME_KEY[state.active_phase]
        )
        data_wait_time_list = timer_recorded_durations.get("data_wait_time")

        # if it's a step hook, we're logging for the previous step, but the data wait time list
        # has already been populated with the current step, so the offset is 2
        data_wait_time_offset = 2 if is_step_end_hook else 1

        if (
            (not iteration_time_list)
            or (not data_wait_time_list)
            or len(data_wait_time_list) < data_wait_time_offset
        ):
            return

        prev_iteration_time = iteration_time_list[-1]
        data_wait_time = data_wait_time_list[-data_wait_time_offset]
        total_time = prev_iteration_time + data_wait_time

        if total_time <= 0:
            return

        for item, num_items in self._throughput_per_batch.items():
            self._logger.log(
                f"{ACTIVE_PHASE_TO_LABEL_PREFIX[state.active_phase]}: {item} per second (step granularity)",
                num_items / total_time,
                step_logging_for,
            )

    def _log_for_epoch(
        self,
        state: State,
        *,
        epoch_logging_for: int,
    ) -> None:

        # Avoid key errors if active phase is not set. This may happen if we restore
        # from an intra-epoch checkpoint.
        if (
            state.active_phase not in self._epoch_start_times
            or state.active_phase not in self._steps_in_epoch
        ):
            logger.warning(
                f"Missing troughput data for epoch {epoch_logging_for}, phase {state.active_phase}. Ommiting troughput logging."
            )
            return

        time_since_epoch_start = (
            time.perf_counter() - self._epoch_start_times[state.active_phase]
        )

        steps_in_epoch = self._steps_in_epoch[state.active_phase]
        if steps_in_epoch <= 0:
            return

        for item, num_items in self._throughput_per_batch.items():
            self._logger.log(
                f"{ACTIVE_PHASE_TO_LABEL_PREFIX[state.active_phase]}: {item} per second (epoch granularity)",
                (num_items * steps_in_epoch) / time_since_epoch_start,
                epoch_logging_for,
            )

        self._steps_in_epoch[state.active_phase] = 0
