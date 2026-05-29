#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import unittest
from typing import Tuple
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from torchtnt.framework._test_utils import DummyFitUnit, generate_random_dataloader
from torchtnt.framework.callback import Callback
from torchtnt.framework.fit import fit
from torchtnt.framework.state import ActivePhase, State
from torchtnt.framework.unit import EvalUnit, TrainUnit, TTrainUnit
from torchtnt.utils.timer import Timer
from torchtnt.utils.version import is_torch_version_geq


class FitTest(unittest.TestCase):
    TORCH_VERSION_GEQ_2_5_0: bool = is_torch_version_geq("2.5.0")

    def test_fit_evaluate_every_n_epochs(self) -> None:
        """
        Test fit entry point with evaluate_every_n_epochs=1
        """
        input_dim = 2
        train_dataset_len = 8
        eval_dataset_len = 4
        batch_size = 2
        max_epochs = 3
        evaluate_every_n_epochs = 1
        expected_train_steps_per_epoch = train_dataset_len / batch_size
        expected_eval_steps_per_epoch = eval_dataset_len / batch_size
        expected_num_evaluate_calls = max_epochs / evaluate_every_n_epochs

        my_unit = DummyFitUnit(input_dim=input_dim)

        train_dataloader = generate_random_dataloader(
            train_dataset_len, input_dim, batch_size
        )
        eval_dataloader = generate_random_dataloader(
            eval_dataset_len, input_dim, batch_size
        )

        fit(
            my_unit,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            max_epochs=max_epochs,
            evaluate_every_n_epochs=evaluate_every_n_epochs,
        )

        self.assertEqual(my_unit.train_progress.num_epochs_completed, max_epochs)
        self.assertEqual(my_unit.train_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            my_unit.train_progress.num_steps_completed,
            max_epochs * expected_train_steps_per_epoch,
        )

        self.assertEqual(
            my_unit.eval_progress.num_epochs_completed,
            expected_num_evaluate_calls,
        )
        self.assertEqual(my_unit.eval_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            my_unit.eval_progress.num_steps_completed,
            max_epochs * expected_eval_steps_per_epoch,
        )

    def test_fit_evaluate_every_n_steps(self) -> None:
        """
        Test fit entry point with evaluate_every_n_steps=5
        """
        input_dim = 2
        train_dataset_len = 16
        eval_dataset_len = 4
        batch_size = 2
        max_epochs = 3
        evaluate_every_n_steps = 5
        expected_train_steps_per_epoch = train_dataset_len / batch_size
        expected_total_train_steps = expected_train_steps_per_epoch * max_epochs
        expected_eval_steps_per_epoch = eval_dataset_len / batch_size
        expected_num_evaluate_calls = math.floor(
            expected_total_train_steps / evaluate_every_n_steps
        )

        my_unit = DummyFitUnit(input_dim=input_dim)

        train_dataloader = generate_random_dataloader(
            train_dataset_len, input_dim, batch_size
        )
        eval_dataloader = generate_random_dataloader(
            eval_dataset_len, input_dim, batch_size
        )

        fit(
            my_unit,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            max_epochs=max_epochs,
            evaluate_every_n_epochs=None,
            evaluate_every_n_steps=evaluate_every_n_steps,
        )

        self.assertEqual(my_unit.train_progress.num_epochs_completed, max_epochs)
        self.assertEqual(my_unit.train_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            my_unit.train_progress.num_steps_completed,
            max_epochs * expected_train_steps_per_epoch,
        )

        self.assertEqual(
            my_unit.eval_progress.num_epochs_completed,
            expected_num_evaluate_calls,
        )
        self.assertEqual(my_unit.eval_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            my_unit.eval_progress.num_steps_completed,
            expected_num_evaluate_calls * expected_eval_steps_per_epoch,
        )

    def test_fit_stop(self) -> None:
        Batch = Tuple[torch.Tensor, torch.Tensor]

        class FitStop(TrainUnit[Batch], EvalUnit[Batch]):
            def __init__(self, input_dim: int, steps_before_stopping: int) -> None:
                super().__init__()
                # initialize module, loss_fn, & optimizer
                self.module = nn.Linear(input_dim, 2)
                self.loss_fn = nn.CrossEntropyLoss()
                self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.01)
                self.steps_processed = 0
                self.steps_before_stopping = steps_before_stopping

            def train_step(
                self, state: State, data: Batch
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                inputs, targets = data

                outputs = self.module(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                assert state.train_state
                if (
                    my_unit.train_progress.num_steps_completed_in_epoch + 1
                    == self.steps_before_stopping
                ):
                    state.stop()

                self.steps_processed += 1
                return loss, outputs

            def eval_step(
                self, state: State, data: Batch
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                inputs, targets = data
                outputs = self.module(inputs)
                loss = self.loss_fn(outputs, targets)
                self.steps_processed += 1
                return loss, outputs

        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 3
        max_steps_per_epoch = 4
        steps_before_stopping = 2

        my_unit = FitStop(
            input_dim=input_dim, steps_before_stopping=steps_before_stopping
        )
        train_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)
        eval_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)

        fit(
            my_unit,
            train_dataloader=train_dl,
            eval_dataloader=eval_dl,
            max_epochs=max_epochs,
            max_train_steps_per_epoch=max_steps_per_epoch,
        )

        self.assertEqual(my_unit.train_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.train_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            my_unit.steps_processed, my_unit.train_progress.num_steps_completed
        )
        self.assertEqual(my_unit.steps_processed, steps_before_stopping)
        self.assertEqual(my_unit.eval_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.eval_progress.num_steps_completed, 0)
        self.assertEqual(my_unit.eval_progress.num_steps_completed_in_epoch, 0)

    def test_fit_max_steps(self) -> None:
        max_steps = 3
        input_dim = 2
        dataset_len = 8
        batch_size = 2
        expected_eval_steps_per_epoch = dataset_len / batch_size

        my_unit = DummyFitUnit(2)
        train_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)
        eval_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)
        fit(
            my_unit,
            train_dataloader=train_dl,
            eval_dataloader=eval_dl,
            max_steps=max_steps,
        )

        self.assertEqual(my_unit.train_progress.num_steps_completed, max_steps)
        self.assertEqual(
            my_unit.eval_progress.num_steps_completed, expected_eval_steps_per_epoch
        )

    def test_fit_with_callback(self) -> None:
        """
        Test fit entry point with a callback
        """
        input_dim = 2
        train_dataset_len = 10
        eval_dataset_len = 6
        batch_size = 2
        max_epochs = 4
        expected_num_total_train_steps = train_dataset_len / batch_size * max_epochs
        expected_num_total_eval_steps = eval_dataset_len / batch_size * max_epochs

        my_unit = DummyFitUnit(2)
        train_dataloader = generate_random_dataloader(
            train_dataset_len, input_dim, batch_size
        )
        eval_dataloader = generate_random_dataloader(
            eval_dataset_len, input_dim, batch_size
        )

        callback_mock = MagicMock(spec=Callback)

        fit(
            my_unit,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            max_epochs=max_epochs,
            callbacks=[callback_mock],
        )

        self.assertEqual(callback_mock.on_train_start.call_count, 1)
        self.assertEqual(callback_mock.on_train_epoch_start.call_count, max_epochs)
        self.assertEqual(
            callback_mock.on_train_step_start.call_count, expected_num_total_train_steps
        )
        self.assertEqual(
            callback_mock.on_train_step_end.call_count, expected_num_total_train_steps
        )
        self.assertEqual(callback_mock.on_train_epoch_end.call_count, max_epochs)
        self.assertEqual(callback_mock.on_train_end.call_count, 1)

        self.assertEqual(callback_mock.on_eval_start.call_count, max_epochs)
        self.assertEqual(callback_mock.on_eval_epoch_start.call_count, max_epochs)
        self.assertEqual(
            callback_mock.on_eval_step_start.call_count, expected_num_total_eval_steps
        )
        self.assertEqual(
            callback_mock.on_eval_step_end.call_count, expected_num_total_eval_steps
        )
        self.assertEqual(callback_mock.on_eval_epoch_end.call_count, max_epochs)
        self.assertEqual(callback_mock.on_eval_end.call_count, max_epochs)

    def test_fit_active_phase(self) -> None:
        tc = unittest.TestCase()

        class PhaseTestCallback(Callback):
            def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
                tc.assertEqual(state.active_phase, ActivePhase.TRAIN)

            def on_train_end(self, state: State, unit: TTrainUnit) -> None:
                tc.assertEqual(state.active_phase, ActivePhase.TRAIN)

        input_dim = 2
        dataset_len = 8
        batch_size = 2
        evaluate_every_n_steps = 2
        evaluate_every_n_epochs = 1
        max_epochs = 2

        my_unit = DummyFitUnit(input_dim)

        train_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)
        eval_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)
        fit(
            my_unit,
            train_dataloader=train_dl,
            eval_dataloader=eval_dl,
            evaluate_every_n_steps=evaluate_every_n_steps,
            evaluate_every_n_epochs=evaluate_every_n_epochs,
            max_epochs=max_epochs,
            callbacks=[PhaseTestCallback()],
        )

    def test_fit_timing(self) -> None:
        """
        Test timing in fit
        """

        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 1
        max_epochs = 1
        evaluate_every_n_epochs = 1

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        timer = Timer()
        fit(
            DummyFitUnit(input_dim=input_dim),
            train_dataloader=dataloader,
            eval_dataloader=dataloader,
            max_train_steps_per_epoch=max_steps_per_epoch,
            max_epochs=max_epochs,
            evaluate_every_n_epochs=evaluate_every_n_epochs,
            timer=timer,
        )
        self.assertIn("train.next(data_iter)", timer.recorded_durations.keys())
        self.assertIn("evaluate.next(data_iter)", timer.recorded_durations.keys())

    def test_error_message(self) -> None:
        self.maxDiff = None
        with self.assertRaises(ValueError), self.assertLogs(level="INFO") as log:
            fit(
                UnitWithError(),
                train_dataloader=[1, 2, 3, 4],
                eval_dataloader=[1, 2, 3, 4],
                max_steps=10,
                evaluate_every_n_epochs=1,
            )

        self.assertIn(
            "INFO:torchtnt.framework.fit:Exception during fit after the following progress: train "
            "progress: completed epochs: 1, completed steps: 4, completed steps in current epoch: 0. "
            "eval progress: completed epochs: 0, completed steps: 2, completed steps in current epoch: 2.:\nfoo",
            log.output,
        )

    @unittest.skipUnless(TORCH_VERSION_GEQ_2_5_0, "test requires PyTorch 2.5.0+")
    @patch(
        "torch.multiprocessing._get_thread_name", side_effect=["foo", "trainer_main"]
    )
    @patch("torch.multiprocessing._set_thread_name")
    def test_fit_set_thread_name(
        self, mock_set_thread_name: MagicMock, mock_get_thread_name: MagicMock
    ) -> None:
        """
        Test fit entry point with evaluate_every_n_epochs=1
        """
        input_dim = 2
        train_dataset_len = 10
        eval_dataset_len = 10
        batch_size = 1

        my_unit = DummyFitUnit(input_dim=input_dim)

        train_dataloader = generate_random_dataloader(
            train_dataset_len, input_dim, batch_size
        )
        eval_dataloader = generate_random_dataloader(
            eval_dataset_len, input_dim, batch_size
        )

        fit(
            my_unit,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            max_epochs=1,
            evaluate_every_n_epochs=1,
        )
        self.assertEqual(mock_get_thread_name.call_count, 2)
        mock_set_thread_name.assert_called_once()

    def test_fit_resume_mid_eval_runs_pending_eval(self) -> None:
        """
        Regression test for the ``tnt_fit_skips_eval_on_resume`` bug.

        Simulates a fit job that was preempted mid-eval by pre-populating
        ``train_progress.num_epochs_completed`` to ``max_epochs`` while
        leaving ``eval_progress.num_epochs_completed`` one step behind. The
        outer fit loop's ``_is_done`` check would normally exit immediately
        in this state; the pre-loop catch-up in ``_train_impl`` should
        detect the pending eval and run it to completion before exiting.
        """
        input_dim = 2
        train_dataset_len = 8
        eval_dataset_len = 4
        batch_size = 2
        max_epochs = 1
        expected_eval_steps_per_epoch = eval_dataset_len / batch_size

        my_unit = DummyFitUnit(input_dim=input_dim)

        # Simulate state restored from a mid-eval EPHEMERAL_PREEMPTION
        # checkpoint: train completed for the (only) epoch, eval did not.
        my_unit.train_progress._num_epochs_completed = max_epochs
        my_unit.train_progress._num_steps_completed = (
            train_dataset_len // batch_size
        ) * max_epochs

        train_dataloader = generate_random_dataloader(
            train_dataset_len, input_dim, batch_size
        )
        eval_dataloader = generate_random_dataloader(
            eval_dataset_len, input_dim, batch_size
        )

        # Spy on eval_step to confirm eval actually ran.
        with patch.object(
            my_unit, "eval_step", wraps=my_unit.eval_step
        ) as mock_eval_step:
            fit(
                my_unit,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_epochs=max_epochs,
                evaluate_every_n_epochs=1,
            )

        # Eval should have run to completion despite ``_is_done`` returning
        # True at loop entry.
        self.assertEqual(mock_eval_step.call_count, int(expected_eval_steps_per_epoch))
        self.assertEqual(my_unit.eval_progress.num_epochs_completed, 1)
        # Train should NOT have re-run; the catch-up runs eval only.
        self.assertEqual(my_unit.train_progress.num_epochs_completed, max_epochs)


class UnitWithError(TrainUnit[int], EvalUnit[int]):
    def train_step(self, state: State, data: int) -> None:
        pass

    def eval_step(self, state: State, data: int) -> None:
        if self.eval_progress.num_steps_completed == 2:
            raise ValueError("foo")
