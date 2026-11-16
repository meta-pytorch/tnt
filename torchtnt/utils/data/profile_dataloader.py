#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Iterable, Optional

import torch
from torch.profiler import record_function
from torchtnt.utils.device import copy_data_to_device
from torchtnt.utils.timer import Timer, TimerProtocol

_log: logging.Logger = logging.getLogger(__name__)


def profile_dataloader(
    dataloader: Iterable[object],
    profiler: torch.profiler.profile,
    *,
    max_steps: Optional[int] = None,
    timer: Optional[TimerProtocol] = None,
    device: Optional[torch.device] = None,
) -> TimerProtocol:
    """
    A helper function that profiles the dataloader iterations.

    Args:
        dataloader: dataloader to be profiled.
        profiler: PyTorch profiler to be used. The profiler is only stepped, so it is the responsibility of the caller to start/stop the profiler.
        max_steps (optional): maximum number of steps to run for. If not set, the dataloader will run until its iterator is exhausted.
        timer (optional): timer to be used to track duration.
        device (optional): device to copy the data to. If set, this function will profile copying data to device.
    """
    timer = timer if timer is not None else Timer(cuda_sync=False)
    with timer.time("iter(dataloader)"), record_function("iter(dataloader)"):
        data_iter = iter(dataloader)

    # If max_steps is not set, run until the dataloader is exhausted
    steps_completed = 0

    while max_steps is None or (steps_completed < max_steps):
        try:
            with timer.time("next(iter)"), record_function("next(iter)"):
                data = next(data_iter)

            if device is not None:
                with timer.time("copy_data_to_device"), record_function(
                    "copy_data_to_device"
                ):
                    data = copy_data_to_device(data, device)

            steps_completed += 1
            if profiler:
                profiler.step()
        except StopIteration:
            break

    return timer
