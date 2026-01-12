# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import cast, Union

from pyre_extensions import none_throws
from torch.utils.tensorboard import SummaryWriter
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit
from torchtnt.utils.distributed import rank_zero_fn
from torchtnt.utils.loggers.logger import MetricLogger
from torchtnt.utils.timer import TimerProtocol


class TimeWaitForBatchLogger(Callback):
    """
    A callback which logs time wait for batch as scalars to a MetricLogger.

    Args:
        logger: Either a subclass of :class:`torchtnt.utils.loggers.logger.MetricLogger`
            or a :class:`torch.utils.tensorboard.SummaryWriter` instance.
        log_every_n_steps: an int to control the log frequency. Default is 1.
        warmup_steps: an int to control the number of warmup steps. We will start logging only after the amount of warmup steps were completed. Default is 0.
    """

    def __init__(
        self,
        logger: Union[MetricLogger, SummaryWriter],
        *,
        log_every_n_steps: int = 1,
        warmup_steps: int = 0,
    ) -> None:
        self._logger = logger
        if log_every_n_steps < 1:
            raise ValueError(
                f"log_every_n_steps must be at least 1, got {log_every_n_steps}"
            )
        self._log_every_n_steps = log_every_n_steps

        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be at least 0, got {warmup_steps}")
        self._warmup_steps = warmup_steps

    @rank_zero_fn
    def _log_step_metrics(
        self,
        *,
        timer: TimerProtocol,
        label: str,
        step: int,
    ) -> None:
        if step <= self._warmup_steps:
            return

        if step % self._log_every_n_steps != 0:
            return

        data_wait_time_list = timer.recorded_durations.get("data_wait_time")
        if not data_wait_time_list:
            return

        if isinstance(self._logger, SummaryWriter):
            self._logger.add_scalar(
                label,
                data_wait_time_list[-1],
                step,
            )
        else:
            cast(MetricLogger, self._logger).log(
                label,
                data_wait_time_list[-1],
                step,
            )

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        self._log_step_metrics(
            timer=none_throws(state.train_state).iteration_timer,
            label="Time Wait For Batch (Train)",
            step=unit.train_progress.num_steps_completed,
        )

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        self._log_step_metrics(
            timer=none_throws(state.eval_state).iteration_timer,
            label="Time Wait For Batch (Eval)",
            step=unit.eval_progress.num_steps_completed,
        )

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        self._log_step_metrics(
            timer=none_throws(state.predict_state).iteration_timer,
            label="Time Wait For Batch (Predict)",
            step=unit.predict_progress.num_steps_completed,
        )
