# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import re
import time
from datetime import datetime, timedelta
from typing import Literal, Optional, Union

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TTrainUnit
from torchtnt.utils.distributed import get_global_rank, sync_bool
from torchtnt.utils.rank_zero_log import rank_zero_info


class TimeLimitInterrupter(Callback):
    """
    This callback tracks the time spent in training and stops the training loop when a time limit is reached. It is possible to define a maximum duration for the training job,
    and/or an absolute timestamp limit. At least one of them should be provided. If both are provided, the callback will stop the training loop when the first condition is met.

    Args:
        duration: Optional, the maximum amount of time to spend in training. Can be specified as a string in the form of DD:HH:MM (days, hours, minutes) or as a timedelta.
            For example, to specify 20 hours is "00:20:00".
        timestamp: Optional datetime object indicating the timestamp at which the training should end. The training will be stopped even if the maximum
            job duration has not been reached yet. Object must be timezone aware.
        interval: Can be either "epoch" or "step". Determines whether to check for time limit exceeding on every epoch or step.
        interval_freq: How often to check for time limit exceeding. For example, if interval is "epoch" and interval_freq is 2, then the callback will check every two epochs.

    Raises:
        ValueError:
            - If the duration is not specified as a string in the form of DD:HH:MM or as a timedelta.
            - If the timestamp datetime object is not timezone aware.
            - If both duration and timestamp are None (i.e. at least one must be specified).

    Note:
        This callback uses the global process group to communicate between ranks.

    """

    def __init__(
        self,
        duration: Optional[Union[str, timedelta]] = None,
        timestamp: Optional[datetime] = None,
        interval: Literal["epoch", "step"] = "epoch",
        interval_freq: int = 1,
    ) -> None:
        if not (duration or timestamp):
            raise ValueError(
                "Invalid parameters. Expected at least one of duration or timestamp to be specified."
            )

        if isinstance(duration, str):
            # checks if string matches DD:HH:MM format and is within valid range
            # 00 <= DD <= 99
            # 00 <= HH <= 23
            # 00 <= MM <= 59
            pattern = r"^\d{2}:(0[0-9]|1[0-9]|2[0-3]):([0-5][0-9])$"
            if not re.match(pattern, duration):
                raise ValueError(
                    f"Invalid duration format '{duration}'. Expected format is DD:HH:MM"
                )
            duration_format = duration.strip().split(":")
            duration_format = list(map(int, duration_format))
            duration = timedelta(
                days=duration_format[0],
                hours=duration_format[1],
                minutes=duration_format[2],
            )

        self._duration: Optional[float] = None
        if duration:
            self._duration = duration.total_seconds()

        self._interval = interval
        self._interval_freq = interval_freq

        self._rank: int = get_global_rank()
        self._start_time: float = 0

        self._timestamp = timestamp
        if timestamp and not timestamp.tzinfo:
            raise ValueError(
                "Invalid timestamp. Expected a timezone aware datetime object."
            )

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        if self._rank == 0:
            self._start_time = time.monotonic()

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        if self._interval == "step":
            if unit.train_progress.num_steps_completed % self._interval_freq == 0:
                self._should_stop(state)

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        if self._interval == "epoch":
            if unit.train_progress.num_epochs_completed % self._interval_freq == 0:
                self._should_stop(state)

    def _should_stop(self, state: State) -> None:
        """
        Check the max duration and the max timestamp to determine if training should stop.
        All ranks sync with rank 0 to determine if any of the stop conditions are met.
        If so, indicates the training loop to stop.
        """
        past_timestamp_limit = False
        past_duration_limit = False

        if self._rank == 0:
            if timestamp := self._timestamp:
                past_timestamp_limit = datetime.now().astimezone() >= timestamp

            if duration := self._duration:
                time_elapsed = time.monotonic() - self._start_time
                past_duration_limit = time_elapsed >= duration

        local_should_stop = past_timestamp_limit or past_duration_limit
        global_should_stop = sync_bool(local_should_stop, coherence_mode="rank_zero")

        if global_should_stop:
            reason = ""
            if past_timestamp_limit:
                reason = f"Training timestamp limit {self._timestamp} has been reached."
            elif past_duration_limit:
                reason = (
                    f"Training duration of {self._duration} seconds has exceeded. "
                    f"Time elapsed is {time.monotonic() - self._start_time} seconds."
                )

            rank_zero_info(f"{reason} Stopping training.")
            state.stop()
