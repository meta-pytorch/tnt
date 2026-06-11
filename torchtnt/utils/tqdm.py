#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import logging
from typing import Iterable, Optional, TextIO, Union

from torchtnt.utils.progress import estimated_steps_in_epoch
from tqdm.auto import tqdm

logger: logging.Logger = logging.getLogger(__name__)


def create_progress_bar(
    dataloader: Iterable[object],
    *,
    desc: str,
    num_epochs_completed: int,
    num_steps_completed: int,
    max_steps: Optional[int],
    max_steps_per_epoch: Optional[int],
    mininterval: float | None = None,
    file: Optional[Union[TextIO, io.StringIO]] = None,
) -> tqdm:
    """Constructs a :func:`tqdm` progress bar. The number of steps in an epoch is inferred from the dataloader, num_steps_completed, max_steps and max_steps_per_epoch.

    Args:
        dataloader: an iterable of data, used to infer number of steps in an epoch.
        desc: a description for the progress bar.
        num_epochs_completed: an integer for the number of epochs completed so far int he loop.
        num_steps_completed: an integer for the number of steps completed so far in the loop.
        max_steps: an optional integer for the number of max steps in the loop.
        max_steps_per_epoch: an optional integer for the number of max steps per epoch.
        mininterval: Minimum display update interval (in seconds). If None, use TQDM's default.
        file: specifies where to output the progress messages (default: sys.stderr)
    """
    current_epoch = num_epochs_completed
    total = estimated_steps_in_epoch(
        dataloader,
        num_steps_completed=num_steps_completed,
        max_steps=max_steps,
        max_steps_per_epoch=max_steps_per_epoch,
    )
    kwargs = {}
    if mininterval is not None:
        kwargs["mininterval"] = mininterval
    # pyrefly: ignore [no-matching-overload]
    return tqdm(
        desc=f"{desc} {current_epoch}",
        total=total,
        initial=num_steps_completed,
        bar_format="{l_bar}{bar}{r_bar}\n",
        file=file,
        **kwargs,
    )


def update_progress_bar(
    progress_bar: tqdm, num_steps_completed: int, refresh_rate: int
) -> None:
    """Updates a progress bar to reflect the number of steps completed."""
    if num_steps_completed % refresh_rate == 0:
        progress_bar.update(refresh_rate)


def close_progress_bar(
    progress_bar: tqdm, num_steps_completed: int, refresh_rate: int
) -> None:
    """Updates and closes a progress bar."""
    # complete remaining progress in bar
    progress_bar.update(num_steps_completed % refresh_rate)
    progress_bar.close()
