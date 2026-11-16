#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torchtnt.framework._test_utils import DummyAutoUnit
from torchtnt.framework.callbacks.progress_reporter import ProgressReporter
from torchtnt.framework.state import EntryPoint, State
from torchtnt.utils.distributed import get_global_rank, spawn_multi_process
from torchtnt.utils.progress import Progress


class ProgressReporterTest(unittest.TestCase):
    def test_log_with_rank(self) -> None:
        spawn_multi_process(2, "gloo", self._test_log_with_rank)

    @staticmethod
    def _test_log_with_rank() -> None:
        progress_reporter = ProgressReporter()
        unit = DummyAutoUnit(module=torch.nn.Linear(2, 2))
        unit.train_progress = Progress(
            num_epochs_completed=1,
            num_steps_completed=5,
            num_steps_completed_in_epoch=3,
        )
        unit.eval_progress = Progress(
            num_epochs_completed=2,
            num_steps_completed=15,
            num_steps_completed_in_epoch=7,
        )
        state = State(entry_point=EntryPoint.FIT)
        tc = unittest.TestCase()
        with tc.assertLogs(level="INFO") as log:
            progress_reporter.on_train_end(state, unit)
        tc.assertEqual(
            log.output,
            [
                f"INFO:torchtnt.framework.callbacks.progress_reporter:Progress Reporter: rank {get_global_rank()} at on_train_end. "
                "Train progress: completed epochs: 1, completed steps: 5, completed steps in current epoch: 3. "
                "Eval progress: completed epochs: 2, completed steps: 15, completed steps in current epoch: 7."
            ],
        )
