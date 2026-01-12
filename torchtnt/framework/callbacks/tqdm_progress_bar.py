# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
from typing import Optional, TextIO, Union

from pyre_extensions import none_throws
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit
from torchtnt.utils.distributed import get_global_rank
from torchtnt.utils.tqdm import (
    close_progress_bar,
    create_progress_bar,
    update_progress_bar,
)
from tqdm.auto import tqdm


class TQDMProgressBar(Callback):
    """
    A callback for progress bar visualization in training, evaluation, and prediction.
    It is initialized only on rank 0 in distributed environments.

    Args:
        refresh_rate: Determines at which rate (in number of steps) the progress bars get updated.
        mininterval: Minimum display update interval (in seconds). If None, use TQDM's default.
        file: specifies where to output the progress messages (default: sys.stderr)
    """

    def __init__(
        self,
        refresh_rate: int = 1,
        file: Optional[Union[TextIO, io.StringIO]] = None,
        *,
        mininterval: float | None = None,
    ) -> None:
        self._refresh_rate = refresh_rate
        self._mininterval = mininterval
        self._file = file

        self._train_progress_bar: Optional[tqdm] = None
        self._eval_progress_bar: Optional[tqdm] = None
        self._predict_progress_bar: Optional[tqdm] = None

    def on_train_epoch_start(self, state: State, unit: TTrainUnit) -> None:
        train_state = none_throws(state.train_state)
        if get_global_rank() == 0:
            self._train_progress_bar = create_progress_bar(
                train_state.dataloader,
                desc="Train Epoch",
                num_epochs_completed=unit.train_progress.num_epochs_completed,
                num_steps_completed=unit.train_progress.num_steps_completed_in_epoch,
                max_steps=train_state.max_steps,
                max_steps_per_epoch=train_state.max_steps_per_epoch,
                file=self._file,
                mininterval=self._mininterval,
            )

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        pbar = self._train_progress_bar
        if pbar is not None:
            update_progress_bar(
                pbar,
                unit.train_progress.num_steps_completed_in_epoch,
                self._refresh_rate,
            )

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        pbar = self._train_progress_bar
        if pbar is not None:
            close_progress_bar(
                pbar,
                unit.train_progress.num_steps_completed_in_epoch,
                self._refresh_rate,
            )

    def on_eval_epoch_start(self, state: State, unit: TEvalUnit) -> None:
        eval_state = none_throws(state.eval_state)
        if get_global_rank() == 0:
            self._eval_progress_bar = create_progress_bar(
                eval_state.dataloader,
                desc="Eval Epoch",
                num_epochs_completed=unit.eval_progress.num_epochs_completed,
                num_steps_completed=unit.eval_progress.num_steps_completed_in_epoch,
                max_steps=eval_state.max_steps,
                max_steps_per_epoch=eval_state.max_steps_per_epoch,
                file=self._file,
                mininterval=self._mininterval,
            )

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        pbar = self._eval_progress_bar
        if pbar is not None:
            update_progress_bar(
                pbar,
                unit.eval_progress.num_steps_completed_in_epoch,
                self._refresh_rate,
            )

    def on_eval_epoch_end(self, state: State, unit: TEvalUnit) -> None:
        pbar = self._eval_progress_bar
        if pbar is not None and state.eval_state:
            close_progress_bar(
                pbar,
                unit.eval_progress.num_steps_completed_in_epoch,
                self._refresh_rate,
            )

    def on_predict_epoch_start(self, state: State, unit: TPredictUnit) -> None:
        predict_state = none_throws(state.predict_state)
        if get_global_rank() == 0:
            self._predict_progress_bar = create_progress_bar(
                predict_state.dataloader,
                desc="Predict Epoch",
                num_epochs_completed=unit.predict_progress.num_epochs_completed,
                num_steps_completed=unit.predict_progress.num_steps_completed,
                max_steps=predict_state.max_steps,
                max_steps_per_epoch=predict_state.max_steps_per_epoch,
                file=self._file,
            )

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        pbar = self._predict_progress_bar
        if pbar is not None:
            update_progress_bar(
                pbar,
                unit.predict_progress.num_steps_completed_in_epoch,
                self._refresh_rate,
            )

    def on_predict_epoch_end(self, state: State, unit: TPredictUnit) -> None:
        pbar = self._predict_progress_bar
        if pbar is not None:
            close_progress_bar(
                pbar,
                unit.predict_progress.num_steps_completed_in_epoch,
                self._refresh_rate,
            )
