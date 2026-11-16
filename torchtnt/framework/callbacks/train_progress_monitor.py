# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Union

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TTrainUnit
from torchtnt.utils.loggers.logger import MetricLogger
from torchtnt.utils.progress import Progress


def _write_training_progress(
    train_progress: Progress, loggers: List[MetricLogger]
) -> None:
    if not loggers:
        return

    step = train_progress.num_steps_completed
    epoch = train_progress.num_epochs_completed
    for logger in loggers:
        logger.log("Training steps completed vs epochs", step, epoch)


class TrainProgressMonitor(Callback):
    """
    A callback which logs training progress in terms of steps vs epochs. This is helpful to visualize when the end of data occurs across epochs, especially for iterable datasets.
    This callback writes to the logger at the beginning of training, and at the end of every epoch.

    Args:
        loggers: Either a :class:`torchtnt.loggers.logger.MetricLogger` or
            list of :class:`torchtnt.loggers.logger.MetricLogger`
    """

    def __init__(
        self,
        loggers: Union[MetricLogger, List[MetricLogger]],
    ) -> None:
        if not isinstance(loggers, list):
            loggers = [loggers]
        self._loggers: List[MetricLogger] = loggers

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        _write_training_progress(unit.train_progress, self._loggers)

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        _write_training_progress(unit.train_progress, self._loggers)
