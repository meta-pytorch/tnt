#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest import mock
from unittest.mock import MagicMock

from torchtnt.framework.callbacks.slow_rank_detector import (
    _get_min_max_indices,
    SlowRankDetector,
)
from torchtnt.framework.state import State
from torchtnt.framework.unit import TrainUnit
from torchtnt.utils.distributed import get_global_rank, spawn_multi_process
from torchtnt.utils.loggers.logger import MetricLogger
from torchtnt.utils.progress import Progress
from torchtnt.utils.test_utils import skip_if_not_distributed, skip_if_not_gpu


class SlowRankDetectorTest(unittest.TestCase):

    @skip_if_not_distributed
    @skip_if_not_gpu
    def test_sync_times(self) -> None:
        spawn_multi_process(2, "nccl", self._test_sync_times)

    @staticmethod
    def _test_sync_times() -> None:
        tc = unittest.TestCase()
        rank = get_global_rank()
        logger = MagicMock(spec=MetricLogger)

        with mock.patch("time.perf_counter", return_value=rank + 1), tc.assertLogs(
            level="INFO"
        ) as log:
            slow_rank_detector = SlowRankDetector(logger=logger)
            slow_rank_detector._sync_times(1, 1)
            tc.assertEqual(
                log.output,
                [
                    "INFO:torchtnt.framework.callbacks.slow_rank_detector:Time difference between fastest rank (0: 1.0 sec) and slowest rank (1: 2.0 sec) is 1.0 seconds after 1 epochs and 1 steps."
                ],
            )
            if rank == 0:
                logger.log.assert_called_once_with(
                    "Difference between fastest/slowest rank (seconds)", 1.0, 1
                )
            else:
                logger.log.assert_not_called()

    def test_get_min_max_indices(self) -> None:
        min_index, max_index = _get_min_max_indices([5.0, 2.0, 3.5])
        self.assertEqual(min_index, 1)
        self.assertEqual(max_index, 0)

        min_index, max_index = _get_min_max_indices([1.0])
        self.assertEqual(min_index, 0)
        self.assertEqual(max_index, 0)

        min_index, max_index = _get_min_max_indices([2.0, 3.0, 2.0])
        self.assertEqual(min_index, 0)
        self.assertEqual(max_index, 1)

    def test_invalid_initialization_params(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "At least one of check_every_n_steps or check_every_n_epochs must be specified.",
        ):
            SlowRankDetector(check_every_n_steps=None, check_every_n_epochs=None)

        with self.assertRaisesRegex(
            ValueError,
            "check_every_n_steps must be a positive integer. Value passed is 0",
        ):
            SlowRankDetector(check_every_n_steps=0)

        with self.assertRaisesRegex(
            ValueError,
            "check_every_n_epochs must be a positive integer. Value passed is 0",
        ):
            SlowRankDetector(check_every_n_epochs=0)

    def test_sync_times_frequency(self) -> None:
        slow_rank_detector = SlowRankDetector(
            check_every_n_steps=2, check_every_n_epochs=2
        )
        unit = MagicMock(spec=TrainUnit)
        unit.train_progress = Progress(num_epochs_completed=1, num_steps_completed=1)
        state = MagicMock(spec=State)
        with mock.patch.object(slow_rank_detector, "_sync_times") as sync_times_mock:
            # first step shouldn't trigger time sync
            slow_rank_detector.on_train_step_end(state, unit)
            sync_times_mock.assert_not_called()

            # second step should trigger time sync
            unit.train_progress.increment_step()
            slow_rank_detector.on_train_step_end(state, unit)
            sync_times_mock.assert_called_once()

            # third step shouldn't trigger time sync
            unit.train_progress.increment_step()
            sync_times_mock.reset_mock()
            slow_rank_detector.on_train_step_end(state, unit)
            sync_times_mock.assert_not_called()

            # first epoch shouldn't trigger time sync
            slow_rank_detector.on_train_epoch_end(state, unit)
            sync_times_mock.assert_not_called()

            # second epoch should trigger time sync
            unit.train_progress.increment_epoch()
            slow_rank_detector.on_train_epoch_end(state, unit)
            sync_times_mock.assert_called_once()

            # third epoch shouldn't trigger time sync
            unit.train_progress.increment_epoch()
            sync_times_mock.reset_mock()
            slow_rank_detector.on_train_epoch_end(state, unit)
            sync_times_mock.assert_not_called()
