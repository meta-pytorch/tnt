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


class IterationTimeLogger(Callback):
    """
    A callback which logs iteration times as scalars to TensorBoard.

    Args:
        logger: Either a :class:`torchtnt.loggers.tensorboard.TensorBoardLogger`
            or a :class:`torch.utils.tensorboard.SummaryWriter` instance.
        moving_avg_window: an int to control the moving average window. Default is 1.
        log_every_n_steps: an int to control the log frequency. Default is 1.
        warmup_steps: an int to control the number of warmup steps. We will start logging only after the amount of warmup steps were completed. Default is 0.
    """

    def __init__(
        self,
        logger: Union[MetricLogger, SummaryWriter],
        *,
        moving_avg_window: int = 1,
        log_every_n_steps: int = 1,
        warmup_steps: int = 0,
    ) -> None:
        self._logger = logger

        if moving_avg_window < 1:
            raise ValueError(
                f"moving_avg_window must be at least 1, got {moving_avg_window}"
            )
        self.moving_avg_window = moving_avg_window

        if log_every_n_steps < 1:
            raise ValueError(
                f"log_every_n_steps must be at least 1, got {log_every_n_steps}"
            )
        self.log_every_n_steps = log_every_n_steps

        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be at least 0, got {warmup_steps}")
        self.warmup_steps = warmup_steps

    @rank_zero_fn
    def _log_step_metrics(
        self,
        metric_label: str,
        iteration_timer: TimerProtocol,
        step_logging_for: int,
    ) -> None:
        """
        Helper function to write a timing log message to writer based on how the class
        was configured.

        """
        if step_logging_for <= self.warmup_steps:
            return

        if step_logging_for % self.log_every_n_steps != 0:
            return

        human_metric_names = {
            "train_iteration_time": "Train Iteration Time (seconds)",
            "eval_iteration_time": "Eval Iteration Time (seconds)",
            "predict_iteration_time": "Prediction Iteration Time (seconds)",
        }

        time_list = iteration_timer.recorded_durations.get(metric_label, [])
        if not time_list:
            return

        last_n_values = time_list[-self.moving_avg_window :]
        if isinstance(self._logger, SummaryWriter):
            self._logger.add_scalar(
                human_metric_names[metric_label],
                sum(last_n_values) / len(last_n_values),
                step_logging_for,
            )
        else:
            cast(MetricLogger, self._logger).log(
                human_metric_names[metric_label],
                sum(last_n_values) / len(last_n_values),
                step_logging_for,
            )

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        timer = none_throws(state.train_state).iteration_timer
        self._log_step_metrics(
            "train_iteration_time",
            timer,
            # on_train_step_end happens after the num steps is incremented, but before the timer list is populated,
            # so it logs for step-1
            unit.train_progress.num_steps_completed - 1,
        )

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        self._log_step_metrics(
            "train_iteration_time",
            none_throws(state.train_state).iteration_timer,
            unit.train_progress.num_steps_completed,
        )

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        timer = none_throws(state.eval_state).iteration_timer
        self._log_step_metrics(
            "eval_iteration_time",
            timer,
            # on_eval_step_end happens after the num steps is incremented, but before the timer list is populated,
            # so it logs for step-1
            unit.eval_progress.num_steps_completed - 1,
        )

    def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
        self._log_step_metrics(
            "eval_iteration_time",
            none_throws(state.eval_state).iteration_timer,
            unit.eval_progress.num_steps_completed,
        )

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        timer = none_throws(state.predict_state).iteration_timer
        self._log_step_metrics(
            "predict_iteration_time",
            timer,
            # on_predict_step_end happens after the num steps is incremented, but before the timer list is populated,
            # so it logs for step-1
            unit.predict_progress.num_steps_completed - 1,
        )

    def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
        timer = none_throws(state.predict_state).iteration_timer
        self._log_step_metrics(
            "predict_iteration_time",
            timer,
            unit.predict_progress.num_steps_completed,
        )
