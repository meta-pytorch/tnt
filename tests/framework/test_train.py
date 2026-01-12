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
from torchtnt.framework._test_utils import DummyTrainUnit, generate_random_dataloader
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.train import train
from torchtnt.framework.unit import TrainUnit, TTrainUnit
from torchtnt.utils.timer import Timer


class TrainTest(unittest.TestCase):
    def test_train(self) -> None:
        """
        Test train entry point
        """
        input_dim = 2
        dataset_len = 8
        batch_size = 2
        max_epochs = 3
        expected_steps_per_epoch = dataset_len / batch_size

        my_unit = DummyTrainUnit(input_dim=input_dim)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        train(
            my_unit,
            dataloader,
            max_epochs=max_epochs,
        )

        self.assertEqual(my_unit.train_progress.num_epochs_completed, max_epochs)
        self.assertEqual(my_unit.train_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            my_unit.train_progress.num_steps_completed,
            max_epochs * expected_steps_per_epoch,
        )

    def test_train_max_steps_per_epoch(self) -> None:
        """
        Test train entry point with max_steps_per_epoch
        """
        input_dim = 2
        dataset_len = 8
        batch_size = 2
        max_epochs = 3
        max_steps_per_epoch = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        initial_training_mode = my_unit.module.training

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        train(
            my_unit,
            dataloader,
            max_epochs=max_epochs,
            max_steps_per_epoch=max_steps_per_epoch,
        )

        self.assertEqual(my_unit.train_progress.num_epochs_completed, max_epochs)
        self.assertEqual(my_unit.train_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            my_unit.train_progress.num_steps_completed,
            max_epochs * max_steps_per_epoch,
        )

        self.assertEqual(my_unit.module.training, initial_training_mode)

    def test_train_stop(self) -> None:
        """
        Test train entry point with setting state's `stop()` flag
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 3
        max_steps_per_epoch = 4
        steps_before_stopping = 2

        my_unit = StopTrainUnit(
            input_dim=input_dim, steps_before_stopping=steps_before_stopping
        )
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        train(
            my_unit,
            dataloader,
            max_epochs=max_epochs,
            max_steps_per_epoch=max_steps_per_epoch,
        )

        self.assertEqual(my_unit.train_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.train_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            my_unit.steps_processed, my_unit.train_progress.num_steps_completed
        )
        self.assertEqual(my_unit.steps_processed, steps_before_stopping)

    def test_train_with_callback(self) -> None:
        """
        Test train entry point with a callback
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 6
        max_epochs = 3
        expected_num_total_steps = dataset_len / batch_size * max_epochs

        my_unit = DummyTrainUnit(2)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        callback_mock = MagicMock(spec=Callback)
        train(
            my_unit,
            dataloader,
            max_epochs=max_epochs,
            max_steps_per_epoch=max_steps_per_epoch,
            callbacks=[callback_mock],
        )
        self.assertEqual(callback_mock.on_train_start.call_count, 1)
        self.assertEqual(callback_mock.on_train_epoch_start.call_count, max_epochs)
        self.assertEqual(
            callback_mock.on_train_step_start.call_count, expected_num_total_steps
        )
        self.assertEqual(
            callback_mock.on_train_step_end.call_count, expected_num_total_steps
        )
        self.assertEqual(callback_mock.on_train_epoch_end.call_count, max_epochs)
        self.assertEqual(callback_mock.on_train_end.call_count, 1)

    def test_train_uses_iteration_timer(self) -> None:
        """
        Test train records time in the iteration_timer
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 1
        max_epochs = 1

        my_unit = DummyTrainUnit(2)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        def assertInTest(key: str, mapping: Mapping[str, Any]) -> None:
            self.assertIn(key, mapping)

        class CheckTimerUsedCallback(Callback):
            def on_train_end(self, state: State, unit: TTrainUnit) -> None:
                assertInTest(
                    "data_wait_time",
                    state.train_state.iteration_timer.recorded_durations,
                )
                assertInTest(
                    "train_iteration_time",
                    state.train_state.iteration_timer.recorded_durations,
                )

        check_timer_callback = CheckTimerUsedCallback()

        train(
            my_unit,
            dataloader,
            max_epochs=max_epochs,
            max_steps_per_epoch=max_steps_per_epoch,
            callbacks=[check_timer_callback],
        )

    def test_train_data_iter_step(self) -> None:
        class TrainIteratorUnit(TrainUnit[Iterator[Tuple[torch.Tensor, torch.Tensor]]]):
            def __init__(self, input_dim: int) -> None:
                super().__init__()
                self.module = nn.Linear(input_dim, 2)
                self.loss_fn = nn.CrossEntropyLoss()

            def train_step(
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

        my_unit = TrainIteratorUnit(input_dim=input_dim)
        initial_training_mode = my_unit.module.training

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        train(my_unit, dataloader, max_epochs=1)

        self.assertEqual(my_unit.train_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.train_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(my_unit.train_progress.num_steps_completed, expected_steps)
        self.assertEqual(my_unit.module.training, initial_training_mode)

    def test_train_max_steps(self) -> None:
        max_steps = 3
        input_dim = 2
        dataset_len = 8
        batch_size = 2
        max_epochs = 3
        expected_steps_per_epoch = dataset_len / batch_size

        my_unit = DummyTrainUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        train(my_unit, dataloader, max_epochs=None, max_steps=max_steps)

        self.assertEqual(my_unit.train_progress.num_steps_completed, max_steps)

        # hit max epoch condition before max steps
        my_unit = DummyTrainUnit(input_dim=input_dim)

        train(my_unit, dataloader, max_epochs=max_epochs, max_steps=100000)
        self.assertEqual(my_unit.train_progress.num_epochs_completed, max_epochs)
        self.assertEqual(my_unit.train_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            my_unit.train_progress.num_steps_completed,
            max_epochs * expected_steps_per_epoch,
        )

    def test_train_timing(self) -> None:
        """
        Test timing in train
        """

        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 1
        max_epochs = 1

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        timer = Timer()
        train(
            DummyTrainUnit(input_dim=input_dim),
            dataloader,
            max_steps_per_epoch=max_steps_per_epoch,
            max_epochs=max_epochs,
            timer=timer,
        )
        self.assertIn("train.next(data_iter)", timer.recorded_durations.keys())

    def test_error_message(self) -> None:
        with self.assertRaises(ValueError), self.assertLogs(level="INFO") as log:
            train(TrainUnitWithError(), [1, 2, 3, 4], max_steps=10)

        self.assertIn(
            "INFO:torchtnt.framework.train:Exception during train after the following progress: "
            "completed epochs: 0, completed steps: 2, completed steps in current epoch: 2.:\nfoo",
            log.output,
        )


Batch = Tuple[torch.Tensor, torch.Tensor]


class StopTrainUnit(TrainUnit[Batch]):
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
            self.train_progress.num_steps_completed_in_epoch + 1
            == self.steps_before_stopping
        ):
            state.stop()

        self.steps_processed += 1
        return loss, outputs


class TrainUnitWithError(TrainUnit[Batch]):
    def train_step(self, state: State, data: Batch) -> None:
        if self.train_progress.num_steps_completed == 2:
            raise ValueError("foo")


Batch = Tuple[torch.Tensor, torch.Tensor]
