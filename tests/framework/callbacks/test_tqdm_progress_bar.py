#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from pyre_extensions import none_throws
from torchtnt.framework._test_utils import (
    DummyEvalUnit,
    DummyPredictUnit,
    DummyTrainUnit,
    generate_random_dataloader,
)
from torchtnt.framework.callbacks.tqdm_progress_bar import TQDMProgressBar
from torchtnt.framework.state import EntryPoint, PhaseState, State
from torchtnt.framework.train import train


class TQDMProgressBarTest(unittest.TestCase):
    def test_progress_bar_train(self) -> None:
        """
        Test TQDMProgressBar callback with train entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 1
        expected_total = dataset_len / batch_size

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = State(
            entry_point=EntryPoint.TRAIN,
            train_state=PhaseState(
                dataloader=dataloader,
                max_epochs=max_epochs,
            ),
        )

        my_unit = DummyTrainUnit(2)
        progress_bar = TQDMProgressBar()
        progress_bar.on_train_epoch_start(state, my_unit)
        self.assertEqual(
            none_throws(progress_bar._train_progress_bar).total, expected_total
        )

    def test_progress_bar_train_integration(self) -> None:
        """
        Test TQDMProgressBar callback with train entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 1

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        my_unit = DummyTrainUnit(2)
        progress_bar = TQDMProgressBar()
        train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[progress_bar])

    def test_progress_bar_evaluate(self) -> None:
        """
        Test TQDMProgressBar callback with evaluate entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 1
        expected_total = dataset_len / batch_size

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = State(
            entry_point=EntryPoint.EVALUATE,
            eval_state=PhaseState(
                dataloader=dataloader,
                max_epochs=max_epochs,
            ),
        )

        my_unit = DummyEvalUnit(2)
        progress_bar = TQDMProgressBar()
        progress_bar.on_eval_epoch_start(state, my_unit)
        self.assertEqual(
            none_throws(progress_bar._eval_progress_bar).total, expected_total
        )

    def test_progress_bar_predict(self) -> None:
        """
        Test TQDMProgressBar callback with predict entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 1
        expected_total = dataset_len / batch_size

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = State(
            entry_point=EntryPoint.PREDICT,
            predict_state=PhaseState(
                dataloader=dataloader,
                max_epochs=max_epochs,
            ),
        )

        my_unit = DummyPredictUnit(2)
        progress_bar = TQDMProgressBar()
        progress_bar.on_predict_epoch_start(state, my_unit)
        self.assertEqual(
            none_throws(progress_bar._predict_progress_bar).total, expected_total
        )

    def test_progress_bar_mid_progress(self) -> None:
        """
        Test TQDMProgressBar callback when progress already has occurred (can occur when loading checkpoint)
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 1
        expected_total = dataset_len / batch_size

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = State(
            entry_point=EntryPoint.PREDICT,
            predict_state=PhaseState(
                dataloader=dataloader,
                max_epochs=max_epochs,
            ),
        )
        my_unit = DummyPredictUnit(2)
        my_unit.predict_progress._num_steps_completed = 2
        progress_bar = TQDMProgressBar()
        progress_bar.on_predict_epoch_start(state, my_unit)
        predict_progress_bar = none_throws(progress_bar._predict_progress_bar)
        self.assertEqual(predict_progress_bar.total, expected_total)
        self.assertEqual(predict_progress_bar.n, 2)
