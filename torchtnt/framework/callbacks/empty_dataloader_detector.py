# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TPredictUnit, TTrainUnit

logger: logging.Logger = logging.getLogger(__name__)


class EmptyDataloaderDetectorCallback(Callback):
    """
    A callback that detects consecutive empty epochs and raises an error when a threshold is reached.

    This callback helps identify issues where dataloaders return empty batches, which can cause confusing
    downstream problems that are hard to debug. It implements a fail-fast strategy to surface these issues early.
    """

    def __init__(
        self,
        threshold: int = 2,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be a positive integer")

        self._threshold = threshold
        self._consecutive_empty_train_epochs = 0

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        num_steps = unit.train_progress.num_steps_completed_in_prev_epoch
        epoch_num = unit.train_progress.num_epochs_completed

        if num_steps == 0:
            self._consecutive_empty_train_epochs += 1
            logger.warning(
                f"Empty train epoch detected! Epoch {epoch_num} completed 0 steps. "
                f"Consecutive empty train epochs: {self._consecutive_empty_train_epochs}"
            )

            if self._consecutive_empty_train_epochs >= self._threshold:
                error_msg = (
                    f"Detected {self._consecutive_empty_train_epochs} consecutive empty train epochs, "
                    f"which exceeds the threshold of {self._threshold}. This indicates that the "
                    f"dataloader is returning empty batches, which could be due to an empty "
                    f"training table or infrastructure issues with the dataloader."
                )
                raise RuntimeError(error_msg)
        else:
            self._consecutive_empty_train_epochs = 0

    def on_predict_epoch_end(self, state: State, unit: TPredictUnit) -> None:
        num_steps_in_predict = unit.predict_progress.num_steps_completed_in_prev_epoch
        if num_steps_in_predict == 0:
            raise RuntimeError(
                "Empty predict epoch detected! Epoch completed 0 steps. This could indicate not enough data in your input."
            )
