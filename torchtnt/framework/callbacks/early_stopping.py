# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Literal

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import AppStateMixin, TEvalUnit, TTrainUnit
from torchtnt.utils.distributed import get_global_rank, sync_bool
from torchtnt.utils.early_stop_checker import EarlyStopChecker

logger: logging.Logger = logging.getLogger(__name__)


class EarlyStopping(Callback):
    """
    This callback checks the value of a monitored attribute on a Unit and stops the training if the value does not improve.

    Args:
        monitored_attr: The attribute to monitor on the unit. Must be a float or tensor attribute on the unit.
        early_stop_checker: a :class:`~torchtnt.utils.early_stop_checker.EarlyStopChecker` to use for checking whether to stop early.
        interval: The interval to check the monitored attribute. Must be one of "step" or "epoch".
        phase: The phase to check the monitored attribute. Must be one of "train" or "eval".

    Note:
        If doing distributed training, this callback checks the metric value only on rank 0
    """

    def __init__(
        self,
        monitored_attr: str,
        early_stop_checker: EarlyStopChecker,
        interval: Literal["step", "epoch"] = "epoch",
        phase: Literal["train", "eval"] = "train",
        interval_freq: int = 1,
    ) -> None:
        self._monitored_attr = monitored_attr
        self._esc = early_stop_checker
        self._interval = interval
        self._interval_freq = interval_freq
        self._phase = phase

        self._rank: int = get_global_rank()

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        if (
            self._phase == "train"
            and self._interval == "step"
            and unit.train_progress.num_steps_completed % self._interval_freq == 0
        ):
            self._maybe_stop(state, unit)

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        if (
            self._phase == "train"
            and self._interval == "epoch"
            and unit.train_progress.num_epochs_completed % self._interval_freq == 0
        ):
            self._maybe_stop(state, unit)

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        if (
            self._phase == "eval"
            and self._interval == "step"
            and unit.eval_progress.num_steps_completed % self._interval_freq == 0
        ):
            self._maybe_stop(state, unit)

    def on_eval_epoch_end(self, state: State, unit: TEvalUnit) -> None:
        if (
            self._phase == "eval"
            and self._interval == "epoch"
            and unit.eval_progress.num_epochs_completed % self._interval_freq == 0
        ):
            self._maybe_stop(state, unit)

    def _maybe_stop(self, state: State, unit: AppStateMixin) -> None:
        """
        Checks whether to stop early based on the monitored attribute.

        Args:
            state: the current state of the training loop.
            unit: the current unit.

        Returns:
            True if the training should stop early, False otherwise.
        """

        if self._rank == 0:
            if not hasattr(unit, self._monitored_attr):
                raise RuntimeError(
                    f"Unit does not have attribute '{self._monitored_attr}', unable to read monitored attribute to determine whether to stop early."
                )

            value = getattr(unit, self._monitored_attr)
            should_stop = self._esc.check(value)
        else:
            should_stop = False

        should_stop = sync_bool(should_stop, coherence_mode="rank_zero")
        if should_stop:
            logger.warning("Stopping training early due to early stopping criteria.")
            state.stop()
