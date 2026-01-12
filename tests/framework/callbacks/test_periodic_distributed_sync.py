#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock, patch

from torchtnt.framework._test_utils import DummyEvalUnit, DummyPredictUnit
from torchtnt.framework.callbacks.periodic_distributed_sync import (
    PeriodicDistributedSync,
)
from torchtnt.framework.state import EntryPoint, State


class PeriodicDistributedSyncTest(unittest.TestCase):
    @patch("torchtnt.framework.callbacks.periodic_distributed_sync.barrier")
    def test_frequency_predict(self, barrier_mock: MagicMock) -> None:
        pds = PeriodicDistributedSync(sync_every_n_steps=2)
        unit = DummyPredictUnit(2)
        state = State(entry_point=EntryPoint.PREDICT)
        unit.predict_progress.increment_step()  # 1 step completed
        pds.on_predict_step_end(state, unit)
        barrier_mock.assert_not_called()

        unit.predict_progress.increment_step()  # 2 steps completed
        pds.on_predict_step_end(state, unit)
        barrier_mock.assert_called_once()

    @patch("torchtnt.framework.callbacks.periodic_distributed_sync.barrier")
    def test_frequency_evaluate(self, barrier_mock: MagicMock) -> None:
        pds = PeriodicDistributedSync(sync_every_n_steps=2)
        unit = DummyEvalUnit(2)
        state = State(entry_point=EntryPoint.EVALUATE)
        unit.eval_progress.increment_step()  # 1 step completed
        pds.on_eval_step_end(state, unit)
        barrier_mock.assert_not_called()

        unit.eval_progress.increment_step()  # 2 steps completed
        pds.on_eval_step_end(state, unit)
        barrier_mock.assert_called_once()
