#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Iterator, Mapping, Tuple
from unittest.mock import MagicMock

import torch
from torch import nn
from torchtnt.framework._test_utils import DummyEvalUnit, generate_random_dataloader
from torchtnt.framework.callback import Callback
from torchtnt.framework.evaluate import evaluate
from torchtnt.framework.state import State
from torchtnt.framework.unit import EvalUnit, TEvalUnit
from torchtnt.utils.timer import Timer


class EvaluateTest(unittest.TestCase):
    def test_evaluate(self) -> None:
        """
        Test evaluate entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        expected_steps = dataset_len / batch_size

        my_unit = DummyEvalUnit(input_dim=input_dim)
        initial_training_mode = my_unit.module.training

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        evaluate(my_unit, dataloader)

        self.assertEqual(my_unit.eval_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.eval_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(my_unit.eval_progress.num_steps_completed, expected_steps)
        self.assertEqual(my_unit.module.training, initial_training_mode)

    def test_evaluate_max_steps_per_epoch(self) -> None:
        """
        Test evaluate entry point with max_steps_per_epoch
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 4

        my_unit = DummyEvalUnit(input_dim=input_dim)
        initial_training_mode = my_unit.module.training

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        evaluate(my_unit, dataloader, max_steps_per_epoch=max_steps_per_epoch)

        self.assertEqual(my_unit.eval_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.eval_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(my_unit.eval_progress.num_steps_completed, max_steps_per_epoch)
        self.assertEqual(my_unit.module.training, initial_training_mode)

    def test_evaluate_stop(self) -> None:
        """
        Test evaluate entry point with setting state's `stop()` flag
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 4
        steps_before_stopping = 2

        my_unit = StopEvalUnit(
            input_dim=input_dim, steps_before_stopping=steps_before_stopping
        )
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        evaluate(my_unit, dataloader, max_steps_per_epoch=max_steps_per_epoch)

        self.assertEqual(my_unit.eval_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.eval_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            my_unit.steps_processed, my_unit.eval_progress.num_steps_completed
        )
        self.assertEqual(my_unit.steps_processed, steps_before_stopping)

    def test_evaluate_data_iter_step(self) -> None:
        class EvalIteratorUnit(EvalUnit[Iterator[Tuple[torch.Tensor, torch.Tensor]]]):
            def __init__(self, input_dim: int) -> None:
                super().__init__()
                self.module = nn.Linear(input_dim, 2)
                self.loss_fn = nn.CrossEntropyLoss()

            def eval_step(
                self, state: State, data: Iterator[Tuple[torch.Tensor, torch.Tensor]]
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                batch = next(data)
                inputs, targets = batch

                outputs = self.module(inputs)
                loss = self.loss_fn(outputs, targets)
                return loss, outputs

        input_dim = 2
        dataset_len = 10
        batch_size = 2
        expected_steps = dataset_len / batch_size

        my_unit = EvalIteratorUnit(input_dim=input_dim)
        initial_training_mode = my_unit.module.training

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        evaluate(my_unit, dataloader)

        self.assertEqual(my_unit.eval_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.eval_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(my_unit.eval_progress.num_steps_completed, expected_steps)
        self.assertEqual(my_unit.module.training, initial_training_mode)

    def test_evaluate_with_callback(self) -> None:
        """
        Test evaluate entry point with a callback
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 6
        expected_num_steps = dataset_len / batch_size

        my_unit = DummyEvalUnit(2)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        callback_mock = MagicMock()

        evaluate(
            my_unit,
            dataloader,
            max_steps_per_epoch=max_steps_per_epoch,
            callbacks=[callback_mock],
        )

        self.assertEqual(callback_mock.on_eval_start.call_count, 1)
        self.assertEqual(callback_mock.on_eval_epoch_start.call_count, 1)
        self.assertEqual(
            callback_mock.on_eval_step_start.call_count, expected_num_steps
        )
        self.assertEqual(callback_mock.on_eval_step_end.call_count, expected_num_steps)
        self.assertEqual(callback_mock.on_eval_epoch_end.call_count, 1)
        self.assertEqual(callback_mock.on_eval_end.call_count, 1)

    def test_evaluate_uses_iteration_timer(self) -> None:
        """
        Test evaluate records time in the iteration_timer
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 1

        my_unit = DummyEvalUnit(2)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        def assertInTest(key: str, mapping: Mapping[str, Any]) -> None:
            self.assertIn(key, mapping)

        class CheckTimerUsedCallback(Callback):
            def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
                assertInTest(
                    "data_wait_time",
                    # pyrefly: ignore [missing-attribute]
                    state.eval_state.iteration_timer.recorded_durations,
                )
                assertInTest(
                    "eval_iteration_time",
                    # pyrefly: ignore [missing-attribute]
                    state.eval_state.iteration_timer.recorded_durations,
                )
                assertInTest(
                    "eval_and_data_iteration_time",
                    # pyrefly: ignore [missing-attribute]
                    state.eval_state.iteration_timer.recorded_durations,
                )

        check_timer_callback = CheckTimerUsedCallback()

        evaluate(
            my_unit,
            dataloader,
            max_steps_per_epoch=max_steps_per_epoch,
            callbacks=[check_timer_callback],
        )

    def test_evaluate_timing(self) -> None:
        """
        Test timing in evaluate
        """

        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 2

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        timer = Timer()
        evaluate(
            DummyEvalUnit(input_dim=input_dim),
            dataloader,
            max_steps_per_epoch=max_steps_per_epoch,
            timer=timer,
        )
        self.assertIn("evaluate.next(data_iter)", timer.recorded_durations.keys())

    def test_error_message(self) -> None:
        with self.assertRaises(ValueError), self.assertLogs(level="INFO") as log:
            evaluate(EvalUnitWithError(), [1, 2, 3, 4])

        self.assertIn(
            "INFO:torchtnt.framework.evaluate:Exception during evaluate after the following progress: "
            "completed epochs: 0, completed steps: 1, completed steps in current epoch: 1.:\nfoo",
            log.output,
        )


class StopEvalUnit(EvalUnit[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, input_dim: int, steps_before_stopping: int) -> None:
        super().__init__()
        # initialize module & loss_fn
        self.module = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.steps_processed = 0
        self.steps_before_stopping = steps_before_stopping

    def eval_step(
        self, state: State, data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)

        assert state.eval_state
        if (
            self.eval_progress.num_steps_completed_in_epoch + 1
            == self.steps_before_stopping
        ):
            state.stop()

        self.steps_processed += 1
        return loss, outputs


class EvalUnitWithError(EvalUnit[int]):
    def eval_step(self, state: State, data: int) -> None:
        if self.eval_progress.num_steps_completed == 1:
            raise ValueError("foo")
