#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from torchtnt.framework import ActivePhase
from torchtnt.framework.state import _check_loop_condition, PhaseState
from torchtnt.utils.checkpoint import Phase


class StateTest(unittest.TestCase):
    def test_check_loop_condition(self) -> None:
        var = "foo"
        _check_loop_condition(var, None)
        _check_loop_condition(var, 100)
        with self.assertRaisesRegex(ValueError, f"Invalid value provided for {var}"):
            _check_loop_condition(var, -1)

    def test_phase_state_validation(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Invalid value provided for max_epochs"
        ):
            PhaseState(dataloader=[], max_epochs=-2)
        with self.assertRaisesRegex(ValueError, "Invalid value provided for max_steps"):
            PhaseState(dataloader=[], max_steps=-2)
        with self.assertRaisesRegex(
            ValueError, "Invalid value provided for max_steps_per_epoch"
        ):
            PhaseState(dataloader=[], max_steps_per_epoch=-2)
        with self.assertRaisesRegex(
            ValueError, "Invalid value provided for evaluate_every_n_steps"
        ):
            PhaseState(dataloader=[], evaluate_every_n_steps=-2)
        with self.assertRaisesRegex(
            ValueError, "Invalid value provided for evaluate_every_n_epochs"
        ):
            PhaseState(dataloader=[], evaluate_every_n_epochs=-2)

    def test_active_phase_into_phase(self) -> None:
        active_phase = ActivePhase.TRAIN
        self.assertEqual(active_phase.into_phase(), Phase.TRAIN)

        eval_phase = ActivePhase.EVALUATE
        self.assertEqual(eval_phase.into_phase(), Phase.EVALUATE)

        predict_phase = ActivePhase.PREDICT
        self.assertEqual(predict_phase.into_phase(), Phase.PREDICT)

    def test_active_phase_str(self) -> None:
        active_phase = ActivePhase.TRAIN
        self.assertEqual(str(active_phase), "train")

        eval_phase = ActivePhase.EVALUATE
        self.assertEqual(str(eval_phase), "eval")

        predict_phase = ActivePhase.PREDICT
        self.assertEqual(str(predict_phase), "predict")

    def test_set_evaluate_every_n_steps_or_epochs(self) -> None:
        state = PhaseState(dataloader=[], evaluate_every_n_steps=2)
        state.evaluate_every_n_steps = None
        state.evaluate_every_n_steps = 100
        with self.assertRaisesRegex(
            ValueError, "Invalid value provided for evaluate_every_n_steps"
        ):
            state.evaluate_every_n_steps = -2

        state = PhaseState(dataloader=[], evaluate_every_n_epochs=2)
        state.evaluate_every_n_epochs = None
        state.evaluate_every_n_epochs = 100
        with self.assertRaisesRegex(
            ValueError, "Invalid value provided for evaluate_every_n_epochs"
        ):
            state.evaluate_every_n_epochs = -2
