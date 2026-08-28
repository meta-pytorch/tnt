#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock

import torch
from torchtnt.framework._test_utils import (
    DummyEvalUnit,
    DummyFitTestUnit,
    DummyFitUnit,
    DummyPredictUnit,
    DummyTestUnit,
    DummyTrainUnit,
    generate_random_dataloader,
)
from torchtnt.framework.callbacks.pytorch_profiler import PyTorchProfiler
from torchtnt.framework.evaluate import evaluate
from torchtnt.framework.fit import fit
from torchtnt.framework.predict import predict
from torchtnt.framework.test import test
from torchtnt.framework.train import train


class PyTorchProfilerTest(unittest.TestCase):
    def test_profiler_train(self) -> None:
        """
        Test PytorchProfiler callback with train entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2
        expected_num_total_steps = dataset_len / batch_size * max_epochs

        my_unit = DummyTrainUnit(input_dim)
        profiler_mock = MagicMock(spec=torch.profiler.profile)

        profiler = PyTorchProfiler(profiler=profiler_mock)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[profiler])
        self.assertEqual(profiler_mock.start.call_count, 1)
        self.assertEqual(profiler_mock.step.call_count, expected_num_total_steps)
        self.assertEqual(profiler_mock.stop.call_count, 1)

    def test_profiler_evaluate(self) -> None:
        """
        Test PytorchProfiler callback with evaluate entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        expected_num_total_steps = dataset_len / batch_size

        my_unit = DummyEvalUnit(2)
        profiler_mock = MagicMock(spec=torch.profiler.profile)

        profiler = PyTorchProfiler(profiler=profiler_mock)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        evaluate(my_unit, dataloader, callbacks=[profiler])
        self.assertEqual(profiler_mock.start.call_count, 1)
        self.assertEqual(profiler_mock.step.call_count, expected_num_total_steps)
        self.assertEqual(profiler_mock.stop.call_count, 1)

    def test_profiler_predict(self) -> None:
        """
        Test PytorchProfiler callback with predict entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        expected_num_total_steps = dataset_len / batch_size

        my_unit = DummyPredictUnit(2)
        profiler_mock = MagicMock(spec=torch.profiler.profile)

        profiler = PyTorchProfiler(profiler=profiler_mock)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        predict(my_unit, dataloader, callbacks=[profiler])
        self.assertEqual(profiler_mock.start.call_count, 1)
        self.assertEqual(profiler_mock.step.call_count, expected_num_total_steps)
        self.assertEqual(profiler_mock.stop.call_count, 1)

    def test_profiler_test(self) -> None:
        """
        Test PytorchProfiler callback with test entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        expected_num_total_steps = dataset_len / batch_size

        my_unit = DummyTestUnit(input_dim)
        profiler_mock = MagicMock(spec=torch.profiler.profile)

        profiler = PyTorchProfiler(profiler=profiler_mock)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        test(my_unit, dataloader, callbacks=[profiler])
        self.assertEqual(profiler_mock.start.call_count, 1)
        self.assertEqual(profiler_mock.step.call_count, expected_num_total_steps)
        self.assertEqual(profiler_mock.stop.call_count, 1)

    def test_profiler_fit_steps_on_train_only(self) -> None:
        """
        Test that in fit, the profiler schedule is advanced by train steps only.

        Eval steps must not call step(), otherwise the schedule drifts ahead of
        the train step it is configured against.
        """
        input_dim = 2
        train_dataset_len = 10
        eval_dataset_len = 6
        batch_size = 2
        max_epochs = 2
        expected_num_total_steps = train_dataset_len / batch_size * max_epochs

        my_unit = DummyFitUnit(input_dim)
        profiler_mock = MagicMock(spec=torch.profiler.profile)

        profiler = PyTorchProfiler(profiler=profiler_mock)

        fit(
            my_unit,
            train_dataloader=generate_random_dataloader(
                train_dataset_len, input_dim, batch_size
            ),
            eval_dataloader=generate_random_dataloader(
                eval_dataset_len, input_dim, batch_size
            ),
            max_epochs=max_epochs,
            evaluate_every_n_epochs=1,
            callbacks=[profiler],
        )

        # started once by on_train_start, stopped once by on_train_end
        self.assertEqual(profiler_mock.start.call_count, 1)
        self.assertEqual(profiler_mock.stop.call_count, 1)
        # eval ran, but contributed no steps to the schedule
        self.assertEqual(profiler_mock.step.call_count, expected_num_total_steps)

    def test_profiler_fit_with_test_dataloader(self) -> None:
        """
        Test that in fit, a trailing test phase does not restart the profiler.

        The test phase runs after training completes, by which point on_train_end
        has already stopped the profiler.
        """
        input_dim = 2
        train_dataset_len = 10
        eval_dataset_len = 6
        test_dataset_len = 4
        batch_size = 2
        max_epochs = 2
        expected_num_total_steps = train_dataset_len / batch_size * max_epochs

        my_unit = DummyFitTestUnit(input_dim)
        profiler_mock = MagicMock(spec=torch.profiler.profile)

        profiler = PyTorchProfiler(profiler=profiler_mock)

        fit(
            my_unit,
            train_dataloader=generate_random_dataloader(
                train_dataset_len, input_dim, batch_size
            ),
            eval_dataloader=generate_random_dataloader(
                eval_dataset_len, input_dim, batch_size
            ),
            max_epochs=max_epochs,
            evaluate_every_n_epochs=1,
            test_dataloader=generate_random_dataloader(
                test_dataset_len, input_dim, batch_size
            ),
            callbacks=[profiler],
        )

        self.assertEqual(profiler_mock.start.call_count, 1)
        self.assertEqual(profiler_mock.stop.call_count, 1)
        self.assertEqual(profiler_mock.step.call_count, expected_num_total_steps)
