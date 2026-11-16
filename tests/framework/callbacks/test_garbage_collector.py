#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import gc
import unittest
from unittest import mock
from unittest.mock import MagicMock

from torchtnt.framework._test_utils import (
    DummyEvalUnit,
    DummyFitUnit,
    DummyPredictUnit,
    DummyTrainUnit,
    generate_random_dataloader,
)
from torchtnt.framework.callbacks.garbage_collector import GarbageCollector
from torchtnt.framework.evaluate import evaluate
from torchtnt.framework.fit import fit
from torchtnt.framework.predict import predict
from torchtnt.framework.train import train


class GarbageCollectorTest(unittest.TestCase):
    def test_garbage_collector_call_count_train(self) -> None:
        """
        Test GarbageCollector callback was called correct number of times (with train entry point)
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2
        expected_num_total_steps = dataset_len / batch_size * max_epochs

        my_unit = DummyTrainUnit(2)
        gc_callback_mock = MagicMock(spec=GarbageCollector)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[gc_callback_mock])
        self.assertEqual(gc_callback_mock.on_train_start.call_count, 1)
        self.assertEqual(
            gc_callback_mock.on_train_step_end.call_count, expected_num_total_steps
        )
        self.assertEqual(gc_callback_mock.on_train_end.call_count, 1)

    def test_garbage_collector_enabled_train(self) -> None:
        """
        Test garbage collection is enabled after runs are finished (with train entry point)
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2

        my_unit = DummyTrainUnit(2)
        gc_callback = GarbageCollector(2)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        self.assertTrue(gc.isenabled())
        train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[gc_callback])
        self.assertTrue(gc.isenabled())

    def test_garbage_collector_call_count_evaluate(self) -> None:
        """
        Test GarbageCollector callback was called correct number of times (with evaluate entry point)
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        expected_num_total_steps = dataset_len / batch_size

        my_unit = DummyEvalUnit(2)
        gc_callback_mock = MagicMock(spec=GarbageCollector)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        evaluate(my_unit, dataloader, callbacks=[gc_callback_mock])
        self.assertEqual(gc_callback_mock.on_eval_start.call_count, 1)
        self.assertEqual(
            gc_callback_mock.on_eval_step_end.call_count, expected_num_total_steps
        )
        self.assertEqual(gc_callback_mock.on_eval_end.call_count, 1)

    def test_garbage_collector_enabled_evaluate(self) -> None:
        """
        Test garbage collection is enabled after runs are finished (with evaluate entry point)
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2

        my_unit = DummyEvalUnit(2)
        gc_callback = GarbageCollector(2)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        self.assertTrue(gc.isenabled())
        evaluate(my_unit, dataloader, callbacks=[gc_callback])
        self.assertTrue(gc.isenabled())

    def test_garbage_collector_call_count_predict(self) -> None:
        """
        Test GarbageCollector callback was called correct number of times (with predict entry point)
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        expected_num_total_steps = dataset_len / batch_size

        my_unit = DummyPredictUnit(2)
        gc_callback_mock = MagicMock(spec=GarbageCollector)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        predict(my_unit, dataloader, callbacks=[gc_callback_mock])
        self.assertEqual(gc_callback_mock.on_predict_start.call_count, 1)
        self.assertEqual(
            gc_callback_mock.on_predict_step_end.call_count, expected_num_total_steps
        )
        self.assertEqual(gc_callback_mock.on_predict_end.call_count, 1)

    def test_garbage_collector_enabled_predict(self) -> None:
        """
        Test garbage collection is enabled after runs are finished (with predict entry point)
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2

        my_unit = DummyPredictUnit(2)
        gc_callback = GarbageCollector(2)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        self.assertTrue(gc.isenabled())
        predict(my_unit, dataloader, callbacks=[gc_callback])
        self.assertTrue(gc.isenabled())

    def test_garbage_collector_call_count_fit(self) -> None:
        """
        Test GarbageCollector callback was called correct number of times (with fit entry point)
        """
        input_dim = 2
        train_dataset_len = 10
        eval_dataset_len = 6
        batch_size = 2
        max_epochs = 2
        evaluate_every_n_epochs = 1
        expected_num_total_steps = (
            train_dataset_len / batch_size * max_epochs
            + eval_dataset_len / batch_size * max_epochs
        )
        gc_step_interval = 4

        my_unit = DummyFitUnit(2)
        gc_callback = GarbageCollector(gc_step_interval)

        train_dataloader = generate_random_dataloader(
            train_dataset_len, input_dim, batch_size
        )
        eval_dataloader = generate_random_dataloader(
            eval_dataset_len, input_dim, batch_size
        )

        # we call gc.collect() once for every step with generation=1, and then we call gc.collect()
        expected_num_calls_to_gc_collect = (
            expected_num_total_steps + expected_num_total_steps / gc_step_interval
        )
        with mock.patch(
            "torchtnt.framework.callbacks.garbage_collector.gc.collect"
        ) as gc_collect_mock:
            fit(
                my_unit,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_epochs=max_epochs,
                evaluate_every_n_epochs=evaluate_every_n_epochs,
                callbacks=[gc_callback],
            )
            self.assertEqual(
                gc_collect_mock.call_count, expected_num_calls_to_gc_collect
            )

    def test_garbage_collector_enabled_fit(self) -> None:
        """
        Test garbage collection is enabled after runs are finished (with fit entry point)
        """
        input_dim = 2
        train_dataset_len = 10
        eval_dataset_len = 6
        batch_size = 2
        max_epochs = 2
        evaluate_every_n_epochs = 1

        my_unit = DummyFitUnit(2)
        gc_callback = GarbageCollector(2)

        train_dataloader = generate_random_dataloader(
            train_dataset_len, input_dim, batch_size
        )
        eval_dataloader = generate_random_dataloader(
            eval_dataset_len, input_dim, batch_size
        )

        self.assertTrue(gc.isenabled())
        fit(
            my_unit,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            max_epochs=max_epochs,
            evaluate_every_n_epochs=evaluate_every_n_epochs,
            callbacks=[gc_callback],
        )
        self.assertTrue(gc.isenabled())
