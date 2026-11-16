# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import cast

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import EntryPoint, State
from torchtnt.framework.unit import AppStateMixin, TEvalUnit, TPredictUnit, TTrainUnit
from torchtnt.utils.distributed import get_global_rank

logger: logging.Logger = logging.getLogger(__name__)


class ProgressReporter(Callback):
    """
    A simple callback which logs the progress at each loop start/end, epoch start/end and step start/end.
    This is useful to debug certain issues, for which the root cause might be unequal progress across ranks, for instance NCCL timeouts.
    If used, it's recommended to pass this callback as the first item in the callbacks list.
    """

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_train_start")

    def on_train_epoch_start(self, state: State, unit: TTrainUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_train_epoch_start")

    def on_train_step_start(self, state: State, unit: TTrainUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_train_step_start")

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_train_step_end")

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_train_epoch_end")

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_train_end")

    def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_eval_start")

    def on_eval_epoch_start(self, state: State, unit: TEvalUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_eval_epoch_start")

    def on_eval_step_start(self, state: State, unit: TEvalUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_eval_step_start")

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_eval_step_end")

    def on_eval_epoch_end(self, state: State, unit: TEvalUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_eval_epoch_end")

    def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_eval_end")

    def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_predict_start")

    def on_predict_epoch_start(self, state: State, unit: TPredictUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_predict_epoch_start")

    def on_predict_step_start(self, state: State, unit: TPredictUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_predict_step_start")

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_predict_step_end")

    def on_predict_epoch_end(self, state: State, unit: TPredictUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_predict_epoch_end")

    def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
        self._log_with_rank_and_unit(state, unit, "on_predict_end")

    @classmethod
    def _log_with_rank_and_unit(
        cls, state: State, unit: AppStateMixin, hook: str
    ) -> None:
        output_str = f"Progress Reporter: rank {get_global_rank()} at {hook}."
        if state.entry_point == EntryPoint.TRAIN:
            output_str = f"{output_str} Train progress: {cast(TTrainUnit, unit).train_progress.get_progress_string()}"

        elif state.entry_point == EntryPoint.EVALUATE:
            output_str = f"{output_str} Eval progress: {cast(TEvalUnit, unit).eval_progress.get_progress_string()}"

        elif state.entry_point == EntryPoint.PREDICT:
            output_str = f"{output_str} Predict progress: {cast(TPredictUnit, unit).predict_progress.get_progress_string()}"

        elif state.entry_point == EntryPoint.FIT:
            output_str = f"{output_str} Train progress: {cast(TTrainUnit, unit).train_progress.get_progress_string()} Eval progress: {cast(TEvalUnit, unit).eval_progress.get_progress_string()}"

        else:
            raise ValueError(
                f"State entry point {state.entry_point} is not supported in ProgressReporter"
            )

        logger.info(output_str)
