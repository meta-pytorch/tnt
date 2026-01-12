# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, Literal
from unittest.mock import MagicMock, patch

from torchtnt.framework._test_utils import (
    Batch,
    get_dummy_eval_state,
    get_dummy_train_state,
)
from torchtnt.framework.callbacks.early_stopping import EarlyStopping
from torchtnt.framework.state import State
from torchtnt.framework.unit import EvalUnit, TrainUnit
from torchtnt.utils.early_stop_checker import EarlyStopChecker


class EarlyStoppingTest(unittest.TestCase):
    def test_invalid_attr(self) -> None:
        early_stop_checker = EarlyStopChecker(
            mode="min",
            patience=2,
        )
        esc = EarlyStopping(
            monitored_attr="foo",
            early_stop_checker=early_stop_checker,
        )

        state = get_dummy_train_state()
        unit = MyTrainLossUnit()

        with self.assertRaisesRegex(RuntimeError, "Unit does not have attribute"):
            esc._maybe_stop(state, unit)

    def test_should_stop(self) -> None:
        early_stop_checker = EarlyStopChecker(
            mode="min",
            patience=2,
            min_delta=0.0,
        )
        esc = EarlyStopping(
            monitored_attr="train_loss",
            early_stop_checker=early_stop_checker,
            interval="epoch",
            interval_freq=1,
        )

        state = get_dummy_train_state()
        unit = MyTrainLossUnit()

        # since patience=2
        for _ in range(2):
            esc._maybe_stop(state, unit)
            self.assertFalse(state._should_stop)
        esc._maybe_stop(state, unit)
        self.assertTrue(state._should_stop)

    @patch("torchtnt.framework.callbacks.early_stopping.EarlyStopping._maybe_stop")
    def test_interval(self, _maybe_stop: MagicMock) -> None:
        early_stop_checker = EarlyStopChecker(
            mode="min",
            patience=2,
            min_delta=0.0,
        )

        state = get_dummy_train_state()
        unit = MyTrainLossUnit()

        # to avoid pyre error
        for interval in ("step", "epoch"):
            esc = EarlyStopping(
                monitored_attr="train_loss",
                early_stop_checker=early_stop_checker,
                interval=cast(Literal["step", "epoch"], interval),
            )

            esc.on_train_step_end(state, unit)
            if interval == "step":
                _maybe_stop.assert_called_once()
            else:
                _maybe_stop.assert_not_called()
            _maybe_stop.reset_mock()

            esc.on_train_epoch_end(state, unit)
            if interval == "epoch":
                _maybe_stop.assert_called_once()
            else:
                _maybe_stop.assert_not_called()

    @patch("torchtnt.framework.callbacks.early_stopping.EarlyStopping._maybe_stop")
    def test_interval_freq(self, _maybe_stop: MagicMock) -> None:
        early_stop_checker = EarlyStopChecker(
            mode="min",
            patience=2,
            min_delta=0.0,
        )
        esc = EarlyStopping(
            monitored_attr="train_loss",
            early_stop_checker=early_stop_checker,
            interval="epoch",
            interval_freq=2,
        )

        state = get_dummy_train_state()
        unit = MyTrainLossUnit()

        unit.train_progress.increment_epoch()
        esc.on_train_epoch_end(state, unit)
        _maybe_stop.assert_not_called()
        unit.train_progress.increment_epoch()
        esc.on_train_epoch_end(state, unit)
        _maybe_stop.assert_called_once()

        _maybe_stop.reset_mock()

        esc = EarlyStopping(
            monitored_attr="train_loss",
            early_stop_checker=early_stop_checker,
            interval="step",
            interval_freq=2,
        )

        unit.train_progress.increment_step()
        esc.on_train_step_end(state, unit)
        _maybe_stop.assert_not_called()
        unit.train_progress.increment_step()
        esc.on_train_step_end(state, unit)
        _maybe_stop.assert_called_once()

    @patch("torchtnt.framework.callbacks.early_stopping.EarlyStopping._maybe_stop")
    def test_phase(self, _maybe_stop: MagicMock) -> None:
        early_stop_checker = EarlyStopChecker(
            mode="min",
            patience=2,
            min_delta=0.0,
        )
        esc = EarlyStopping(
            monitored_attr="eval_loss",
            early_stop_checker=early_stop_checker,
            interval="epoch",
            interval_freq=2,
            phase="eval",
        )

        state = get_dummy_eval_state()
        unit = MyEvalLossUnit()

        unit.eval_progress.increment_epoch()
        esc.on_eval_epoch_end(state, unit)
        _maybe_stop.assert_not_called()
        unit.eval_progress.increment_epoch()
        esc.on_eval_epoch_end(state, unit)
        _maybe_stop.assert_called_once()

        _maybe_stop.reset_mock()

        esc = EarlyStopping(
            monitored_attr="eval_loss",
            early_stop_checker=early_stop_checker,
            interval="step",
            interval_freq=2,
            phase="eval",
        )

        unit.eval_progress.increment_step()
        esc.on_eval_step_end(state, unit)
        _maybe_stop.assert_not_called()
        unit.eval_progress.increment_step()
        esc.on_eval_step_end(state, unit)
        _maybe_stop.assert_called_once()


class MyTrainLossUnit(TrainUnit[Batch]):
    def __init__(self) -> None:
        super().__init__()
        self.train_loss = 0.01

    def train_step(self, state: State, data: Batch) -> None:
        return None


class MyEvalLossUnit(EvalUnit[Batch]):
    def __init__(self) -> None:
        super().__init__()
        self.eval_loss = 0.01

    def eval_step(self, state: State, data: Batch) -> None:
        return None
