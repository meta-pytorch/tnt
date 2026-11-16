# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import time
from typing import List, Optional, Tuple

import torch
from torch import distributed as dist
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TTrainUnit
from torchtnt.utils.distributed import all_gather_tensors, get_global_rank
from torchtnt.utils.env import init_from_env
from torchtnt.utils.loggers.logger import MetricLogger

logger: logging.Logger = logging.getLogger(__name__)


class SlowRankDetector(Callback):
    """
    A callback which detects slow ranks every N steps/epochs by comparing the time on each process.
    This is useful to debug ranks which are lagging behind and are likely to cause a NCCL timeout.
    If a logger is passed, the difference between the fastest rank and slowest rank is also reported.

    Args:
        check_every_n_steps: frequency of steps to check for slow ranks.
        check_every_n_epochs: frequency of epochs to check for slow ranks.
        pg: the process group to use for all_gather_tensors. If None, the default process group will be used.
        logger: an optional logger to log time difference.
        device: the device that will be used to store the time as a tensor. If none, the device will be inferred from the environment.

    Note:
        It is recommended to use this callback after you detect a timeout, and to make sure this callback runs before
        the logic triggering timeout (other callback, train_step, etc).
    """

    def __init__(
        self,
        *,
        check_every_n_steps: Optional[int] = 100,
        check_every_n_epochs: Optional[int] = 1,
        pg: Optional[dist.ProcessGroup] = None,
        logger: Optional[MetricLogger] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if not (check_every_n_steps or check_every_n_epochs):
            raise ValueError(
                "At least one of check_every_n_steps or check_every_n_epochs must be specified."
            )

        if check_every_n_steps is not None and check_every_n_steps <= 0:
            raise ValueError(
                f"check_every_n_steps must be a positive integer. Value passed is {check_every_n_steps}"
            )

        if check_every_n_epochs is not None and check_every_n_epochs <= 0:
            raise ValueError(
                f"check_every_n_epochs must be a positive integer. Value passed is {check_every_n_epochs}"
            )

        self._check_every_n_steps = check_every_n_steps
        self._check_every_n_epochs = check_every_n_epochs
        self._pg = pg
        self._logger = logger
        self._device: torch.device = device or init_from_env()
        self._rank: int = get_global_rank()

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        if (
            self._check_every_n_steps is not None
            and unit.train_progress.num_steps_completed % self._check_every_n_steps == 0
        ):
            self._sync_times(
                unit.train_progress.num_epochs_completed,
                unit.train_progress.num_steps_completed,
            )

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        if (
            self._check_every_n_epochs is not None
            and unit.train_progress.num_epochs_completed % self._check_every_n_epochs
            == 0
        ):
            self._sync_times(
                unit.train_progress.num_epochs_completed,
                unit.train_progress.num_steps_completed,
            )

    def _sync_times(self, epochs: int, steps: int) -> None:
        curr_time = time.perf_counter()
        curr_time_tensor = torch.Tensor([curr_time]).to(self._device)
        timings_as_tensor_list = all_gather_tensors(curr_time_tensor, self._pg)
        timings_as_list: List[float] = [
            tensor.item() for tensor in timings_as_tensor_list
        ]
        fastest_rank, slowest_rank = _get_min_max_indices(timings_as_list)
        time_on_fastest_rank = timings_as_list[fastest_rank]
        time_on_slowest_rank = timings_as_list[slowest_rank]
        time_difference = time_on_slowest_rank - time_on_fastest_rank
        logger.info(
            f"""Time difference between fastest rank ({fastest_rank}: {time_on_fastest_rank} sec) and slowest rank ({slowest_rank}: {time_on_slowest_rank} sec) is {time_difference} seconds after {epochs} epochs and {steps} steps."""
        )
        if self._logger and self._rank == 0:
            self._logger.log(
                "Difference between fastest/slowest rank (seconds)",
                time_difference,
                steps,
            )


# instead of taking a dependency on numpy
def _get_min_max_indices(input_list: List[float]) -> Tuple[int, int]:
    min_index = -1
    max_index = -1
    min_value = float("inf")
    max_value = float("-inf")
    for rank, curr_value in enumerate(input_list):
        if curr_value < min_value:
            min_value = curr_value
            min_index = rank
        if curr_value > max_value:
            max_value = curr_value
            max_index = rank

    return min_index, max_index
