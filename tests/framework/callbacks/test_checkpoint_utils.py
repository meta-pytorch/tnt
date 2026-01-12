# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from pyre_extensions import none_throws
from torch import nn
from torchtnt.framework import ActivePhase
from torchtnt.framework._test_utils import (
    DummyAutoUnit,
    DummyEvalUnit,
    DummyFitUnit,
    DummyMeanMetric,
    DummyTrainUnit,
    generate_dummy_stateful_dataloader,
    get_dummy_eval_state,
    get_dummy_fit_state,
    get_dummy_predict_state,
    get_dummy_train_state,
)
from torchtnt.framework.callbacks._checkpoint_utils import (
    _get_epoch,
    _get_step_phase_mapping,
    _prepare_app_state_for_checkpoint,
)
from torchtnt.utils.checkpoint import Phase


class CheckpointUtilsTest(unittest.TestCase):

    def test_get_app_state(self) -> None:

        # Test end-of-epoch checkpoint
        my_unit = DummyTrainUnit(input_dim=2)
        state = get_dummy_train_state()
        app_state = _prepare_app_state_for_checkpoint(state, my_unit, intra_epoch=False)
        self.assertCountEqual(
            app_state.keys(),
            ["module", "optimizer", "loss_fn", "train_progress"],
        )

        # Test train intra-epoch checkpoint
        my_unit = DummyTrainUnit(input_dim=2)
        my_unit.mean_metric = DummyMeanMetric()  # pyre-ignore[16]
        state = get_dummy_train_state()
        stateful_dl = generate_dummy_stateful_dataloader(1, 1, 1)
        state._active_phase = ActivePhase.TRAIN
        none_throws(state.train_state)._dataloader = stateful_dl

        app_state = _prepare_app_state_for_checkpoint(state, my_unit, intra_epoch=True)
        self.assertCountEqual(
            app_state.keys(),
            [
                "module",
                "optimizer",
                "loss_fn",
                "train_progress",
                "train_dataloader",
                "mean_metric",
            ],
        )

        # Test evaluate intra-epoch checkpoint
        my_unit = DummyEvalUnit(input_dim=2)
        my_unit.mean_metric = DummyMeanMetric()  # pyre-ignore[16]
        state = get_dummy_eval_state()
        stateful_dl = generate_dummy_stateful_dataloader(1, 1, 1)
        state._active_phase = ActivePhase.EVALUATE
        none_throws(state.eval_state)._dataloader = stateful_dl

        app_state = _prepare_app_state_for_checkpoint(state, my_unit, intra_epoch=True)
        self.assertCountEqual(
            app_state.keys(),
            [
                "eval_progress",
                "eval_dataloader",
                "mean_metric",
            ],
        )

        # Test evaluate intra-epoch within train epoch on FIT (evaluate_every_n_steps)
        my_unit = DummyFitUnit(input_dim=2)
        my_unit.train_progress.increment_step()  # Simulate at least one step in each phase
        my_unit.eval_progress.increment_step()

        state = get_dummy_fit_state()
        state._active_phase = ActivePhase.EVALUATE

        train_dl = generate_dummy_stateful_dataloader(1, 1, 1)
        eval_dl = generate_dummy_stateful_dataloader(1, 1, 1)
        none_throws(state.train_state)._dataloader = train_dl
        none_throws(state.eval_state)._dataloader = eval_dl

        app_state = _prepare_app_state_for_checkpoint(state, my_unit, intra_epoch=True)
        self.assertCountEqual(
            app_state.keys(),
            [
                "module",
                "optimizer",
                "loss_fn",
                "train_progress",
                "eval_progress",
                "train_dataloader",
                "eval_dataloader",
            ],
        )

    def test_get_step_phase_mapping(self) -> None:
        unit = DummyAutoUnit(module=nn.Linear(2, 2))
        unit.train_progress._num_steps_completed = 5
        unit.eval_progress._num_steps_completed = 7
        unit.predict_progress._num_steps_completed = 9

        fit_state = get_dummy_fit_state()
        self.assertEqual(
            {Phase.TRAIN: 5, Phase.EVALUATE: 7},
            _get_step_phase_mapping(fit_state, unit),
        )

        train_state = get_dummy_train_state()
        self.assertEqual({Phase.TRAIN: 5}, _get_step_phase_mapping(train_state, unit))

        eval_state = get_dummy_eval_state()
        self.assertEqual({Phase.EVALUATE: 7}, _get_step_phase_mapping(eval_state, unit))

        predict_state = get_dummy_predict_state()
        self.assertEqual(
            {Phase.PREDICT: 9}, _get_step_phase_mapping(predict_state, unit)
        )

    def test_get_epoch(self) -> None:
        unit = DummyAutoUnit(module=nn.Linear(2, 2))
        unit.train_progress._num_epochs_completed = 1
        unit.eval_progress._num_epochs_completed = 2
        unit.predict_progress._num_epochs_completed = 3

        fit_state = get_dummy_fit_state()
        self.assertEqual(1, _get_epoch(fit_state, unit))

        train_state = get_dummy_train_state()
        self.assertEqual(1, _get_epoch(train_state, unit))

        eval_state = get_dummy_eval_state()
        self.assertEqual(2, _get_epoch(eval_state, unit))

        predict_state = get_dummy_predict_state()
        self.assertEqual(3, _get_epoch(predict_state, unit))
