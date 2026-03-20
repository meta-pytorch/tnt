#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Iterator, Tuple
from unittest.mock import MagicMock

import torch
from torch import nn
from torchtnt.framework._test_utils import (
    DummyFitTestUnit,
    DummyTestUnit,
    generate_random_dataloader,
)
from torchtnt.framework.callback import Callback
from torchtnt.framework.fit import fit
from torchtnt.framework.state import State
from torchtnt.framework.test import test
from torchtnt.framework.unit import TestUnit
from torchtnt.utils.timer import Timer


class TestTest(unittest.TestCase):
    def test_test_basic(self) -> None:
        """
        Test the test() entry point runs all steps.
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        expected_steps = dataset_len / batch_size

        my_unit = DummyTestUnit(input_dim=input_dim)
        initial_training_mode = my_unit.module.training

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        test(my_unit, dataloader)

        self.assertEqual(my_unit.test_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.test_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(my_unit.test_progress.num_steps_completed, expected_steps)
        self.assertEqual(my_unit.module.training, initial_training_mode)

    def test_test_max_steps_per_epoch(self) -> None:
        """
        Test the test() entry point with max_steps_per_epoch.
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 3

        my_unit = DummyTestUnit(input_dim=input_dim)
        initial_training_mode = my_unit.module.training

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        test(my_unit, dataloader, max_steps_per_epoch=max_steps_per_epoch)

        self.assertEqual(my_unit.test_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.test_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(my_unit.test_progress.num_steps_completed, max_steps_per_epoch)
        self.assertEqual(my_unit.module.training, initial_training_mode)

    def test_test_stop(self) -> None:
        """
        Test the test() entry point with state's stop() flag.
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        steps_before_stopping = 2

        my_unit = StopTestUnit(
            input_dim=input_dim, steps_before_stopping=steps_before_stopping
        )
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        test(my_unit, dataloader)

        self.assertEqual(my_unit.test_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.test_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            my_unit.steps_processed, my_unit.test_progress.num_steps_completed
        )
        self.assertEqual(my_unit.steps_processed, steps_before_stopping)

    def test_test_data_iter_step(self) -> None:
        """
        Test the test() entry point with an iterator-based test_step.
        """

        class TestIteratorUnit(TestUnit[Iterator[Tuple[torch.Tensor, torch.Tensor]]]):
            def __init__(self, input_dim: int) -> None:
                super().__init__()
                self.module = nn.Linear(input_dim, 2)
                self.loss_fn = nn.CrossEntropyLoss()

            def test_step(
                self,
                state: State,
                data: Iterator[Tuple[torch.Tensor, torch.Tensor]],
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

        my_unit = TestIteratorUnit(input_dim=input_dim)
        initial_training_mode = my_unit.module.training

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        test(my_unit, dataloader)

        self.assertEqual(my_unit.test_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.test_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(my_unit.test_progress.num_steps_completed, expected_steps)
        self.assertEqual(my_unit.module.training, initial_training_mode)

    def test_test_with_callback(self) -> None:
        """
        Test the test() entry point with callbacks.
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2

        my_unit = DummyTestUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        callback_mock = MagicMock(spec=Callback)
        test(my_unit, dataloader, callbacks=[callback_mock])

        self.assertEqual(callback_mock.on_test_start.call_count, 1)
        self.assertEqual(callback_mock.on_test_epoch_start.call_count, 1)
        self.assertEqual(callback_mock.on_test_epoch_end.call_count, 1)
        self.assertEqual(callback_mock.on_test_end.call_count, 1)
        self.assertEqual(
            callback_mock.on_test_step_start.call_count,
            dataset_len / batch_size,
        )
        self.assertEqual(
            callback_mock.on_test_step_end.call_count,
            dataset_len / batch_size,
        )

    def test_test_with_timer(self) -> None:
        """
        Test the test() entry point with a timer.
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2

        my_unit = DummyTestUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        timer = Timer()
        test(my_unit, dataloader, timer=timer)

        self.assertEqual(my_unit.test_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.test_progress.num_steps_completed, 5)

    def test_test_no_grad(self) -> None:
        """
        Test that test_step runs under torch.no_grad() context.
        """

        class GradCheckTestUnit(TestUnit[Tuple[torch.Tensor, torch.Tensor]]):
            def __init__(self) -> None:
                super().__init__()
                self.module = nn.Linear(2, 2)
                self.grad_enabled: bool = True

            def test_step(
                self, state: State, data: Tuple[torch.Tensor, torch.Tensor]
            ) -> Any:
                self.grad_enabled = torch.is_grad_enabled()
                return None

        my_unit = GradCheckTestUnit()
        dataloader = [(torch.randn(2, 2), torch.randint(0, 2, (2,)))]
        test(my_unit, dataloader)

        self.assertFalse(my_unit.grad_enabled)

    def test_test_exception_handling(self) -> None:
        """
        Test that exceptions during test are properly handled.
        """

        class FailingTestUnit(TestUnit[Tuple[torch.Tensor, torch.Tensor]]):
            def __init__(self) -> None:
                super().__init__()
                self.module = nn.Linear(2, 2)

            def test_step(
                self, state: State, data: Tuple[torch.Tensor, torch.Tensor]
            ) -> Any:
                raise RuntimeError("test failure")

        my_unit = FailingTestUnit()
        callback_mock = MagicMock(spec=Callback)
        dataloader = [(torch.randn(2, 2), torch.randint(0, 2, (2,)))]

        with self.assertRaises(RuntimeError):
            test(my_unit, dataloader, callbacks=[callback_mock])

        callback_mock.on_exception.assert_called_once()

    def test_test_empty_dataloader(self) -> None:
        """
        Test the test() entry point with an empty dataloader.
        """
        my_unit = DummyTestUnit(input_dim=2)
        empty_dataloader: list[Tuple[torch.Tensor, torch.Tensor]] = []
        test(my_unit, empty_dataloader)

        self.assertEqual(my_unit.test_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.test_progress.num_steps_completed, 0)

    def test_fit_with_test_dataloader(self) -> None:
        """
        Test fit() runs test phase after training when test_dataloader is provided.
        """
        input_dim = 2
        dataset_len = 8
        batch_size = 2

        my_unit = DummyFitTestUnit(input_dim=input_dim)

        train_dataloader = generate_random_dataloader(
            dataset_len, input_dim, batch_size
        )
        eval_dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        test_dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        fit(
            my_unit,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            max_epochs=1,
            test_dataloader=test_dataloader,
        )

        # Verify training completed
        self.assertEqual(my_unit.train_progress.num_epochs_completed, 1)
        # Verify test phase ran
        self.assertEqual(my_unit.test_progress.num_epochs_completed, 1)
        self.assertEqual(
            my_unit.test_progress.num_steps_completed, dataset_len / batch_size
        )

    def test_fit_without_test_dataloader(self) -> None:
        """
        Test fit() still works without test_dataloader (backward compat).
        """
        input_dim = 2
        dataset_len = 8
        batch_size = 2

        my_unit = DummyFitTestUnit(input_dim=input_dim)

        train_dataloader = generate_random_dataloader(
            dataset_len, input_dim, batch_size
        )
        eval_dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        fit(
            my_unit,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            max_epochs=1,
        )

        # Verify training completed
        self.assertEqual(my_unit.train_progress.num_epochs_completed, 1)
        # Verify test phase did NOT run
        self.assertEqual(my_unit.test_progress.num_epochs_completed, 0)
        self.assertEqual(my_unit.test_progress.num_steps_completed, 0)


class StopTestUnit(TestUnit[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, input_dim: int, steps_before_stopping: int) -> None:
        super().__init__()
        self.module = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.steps_before_stopping = steps_before_stopping
        self.steps_processed = 0

    def test_step(
        self, state: State, data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data
        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        self.steps_processed += 1
        if self.steps_processed >= self.steps_before_stopping:
            state.stop()
        return loss, outputs


if __name__ == "__main__":
    unittest.main()
