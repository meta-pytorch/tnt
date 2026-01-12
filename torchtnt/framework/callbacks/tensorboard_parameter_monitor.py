# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, Optional, Union

import torch
from torch.utils.tensorboard import SummaryWriter
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TTrainUnit
from torchtnt.utils.loggers.tensorboard import TensorBoardLogger


def _write_histogram_parameters(
    summary_writer: SummaryWriter, modules: Dict[str, torch.nn.Module], step: int
) -> None:
    for module_name, module in modules.items():
        for param_name, parameter in module.named_parameters():
            summary_writer.add_histogram(
                f"Parameters/{module_name}/{param_name}",
                parameter,
                global_step=step,
            )


class TensorBoardParameterMonitor(Callback):
    """
    A callback which logs module parameters as histograms to TensorBoard.
    https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter

    Args:
        logger: Either a :class:`torchtnt.loggers.tensorboard.TensorBoardLogger`
            or a :class:`torch.utils.tensorboard.SummaryWriter` instance.
    """

    def __init__(self, logger: Union[TensorBoardLogger, SummaryWriter]) -> None:
        if isinstance(logger, TensorBoardLogger):
            logger = logger.writer
        self._writer: Optional[SummaryWriter] = logger

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        writer = self._writer
        if not writer:
            return

        step = unit.train_progress.num_steps_completed
        modules = unit.tracked_modules()
        _write_histogram_parameters(writer, modules, step)
