#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import call, MagicMock

import torch
from pyre_extensions import none_throws
from torch.utils.tensorboard import SummaryWriter
from torchtnt.framework._callback_handler import CallbackHandler
from torchtnt.framework._test_utils import (
    DummyAutoUnit,
    DummyEvalUnit,
    DummyPredictUnit,
    DummyTrainUnit,
    generate_random_dataloader,
)
from torchtnt.framework.callbacks.iteration_time_logger import IterationTimeLogger
from torchtnt.framework.state import EntryPoint, PhaseState, State
from torchtnt.framework.train import _train_impl, train
from torchtnt.utils.loggers.logger import MetricLogger
from torchtnt.utils.timer import Timer


class IterationTimeLoggerTest(unittest.TestCase):
    def test_iteration_time_logger_test_on_train_step_end(self) -> None:
        logger = MagicMock(spec=MetricLogger)
        state = MagicMock(spec=State)

        # Test that the recorded times are tracked separately and that we properly
        # handle when there are no values for a given metric.
        recorded_durations = {
            "train_iteration_time": [1, 3, 5, 7, 9],
            "eval_iteration_time": [],
            "predict_iteration_time": [11, 13, 15, 17, 19],
        }
        state.train_state.iteration_timer.recorded_durations = recorded_durations.copy()
        state.eval_state.iteration_timer.recorded_durations = recorded_durations.copy()
        state.predict_state.iteration_timer.recorded_durations = (
            recorded_durations.copy()
        )

        callback = IterationTimeLogger(logger=logger, moving_avg_window=4)

        # Set up some dummy units for invoking the callback.
        train_unit = DummyTrainUnit(input_dim=2)
        train_unit.train_progress.increment_step()
        train_unit.train_progress.increment_step()

        eval_unit = DummyEvalUnit(input_dim=2)
        eval_unit.eval_progress.increment_step()
        eval_unit.eval_progress.increment_step()

        predict_unit = DummyPredictUnit(input_dim=2)
        predict_unit.predict_progress.increment_step()
        predict_unit.predict_progress.increment_step()

        # Invoke the callback for each step type we are tracking iteration time for.
        callback = IterationTimeLogger(logger=logger, moving_avg_window=4)
        callback.on_train_step_end(state, train_unit)
        callback.on_eval_step_end(state, eval_unit)
        callback.on_predict_step_end(state, predict_unit)

        logger.log.assert_has_calls(
            [
                call(
                    "Train Iteration Time (seconds)",
                    6.0,  # the average of the last 4 numbers is 6
                    1,  # at on_train_step_end we report for step-1, we incremented twice so value should be 1
                ),
                call(
                    "Prediction Iteration Time (seconds)",
                    16.0,  # the average of the last 4 numbers is 16
                    1,  # at on_predict_step_end we report for step-1, we incremented twice so value should be 1
                ),
            ]
        )

    def test_with_train_epoch(self) -> None:
        """
        Test IterationTimeLogger callback with train entry point
        """

        my_unit = DummyTrainUnit(input_dim=2)
        logger = MagicMock(spec=MetricLogger)
        callback = IterationTimeLogger(logger, moving_avg_window=1, log_every_n_steps=3)
        dataloader = generate_random_dataloader(
            num_samples=12, input_dim=2, batch_size=2
        )
        train(my_unit, dataloader, max_epochs=2, callbacks=[callback])
        # 2 epochs, 6 iterations each, logging every third step
        self.assertEqual(logger.log.call_count, 4)

    def test_comparing_step_logging_time(self) -> None:
        """
        Test IterationTimeLogger callback and compare reported time to collected time
        """

        my_auto_unit = DummyAutoUnit(module=torch.nn.Linear(2, 2))
        logger = MagicMock(spec=MetricLogger)
        iteration_time_logger = IterationTimeLogger(
            logger, moving_avg_window=1, log_every_n_steps=1
        )
        dataloader = generate_random_dataloader(
            num_samples=8, input_dim=2, batch_size=2
        )
        state = State(
            entry_point=EntryPoint.FIT,
            train_state=PhaseState(
                dataloader=dataloader,
                max_epochs=2,
                max_steps_per_epoch=2,
            ),
            eval_state=PhaseState(
                dataloader=dataloader,
                max_steps_per_epoch=2,
                evaluate_every_n_epochs=1,
            ),
        )

        # we want to be able to compare the logging value to the state, so we need to create state manually and
        # call _train_impl. This would have been similar to calling fit() and getting the state as a ret value

        _train_impl(state, my_auto_unit, CallbackHandler([iteration_time_logger]))
        train_iteration_timer = none_throws(
            state.train_state
        ).iteration_timer.recorded_durations["train_iteration_time"]
        eval_iteration_timer = none_throws(
            state.eval_state
        ).iteration_timer.recorded_durations["eval_iteration_time"]

        expected_training_iteration_time_calls = [
            call("Train Iteration Time (seconds)", train_iteration_timer[i], i + 1)
            for i in range(4)
        ]
        expected_eval_iteration_time_calls = [
            call("Eval Iteration Time (seconds)", eval_iteration_timer[i], i + 1)
            for i in range(4)
        ]

        logger.log.assert_has_calls(
            expected_training_iteration_time_calls + expected_eval_iteration_time_calls,
            any_order=True,
        )

    def test_with_summary_writer(self) -> None:
        """
        Test IterationTimeLogger callback with train entry point and SummaryWriter
        """

        my_unit = DummyTrainUnit(input_dim=2)
        logger = MagicMock(spec=SummaryWriter)
        callback = IterationTimeLogger(logger, moving_avg_window=1, log_every_n_steps=3)
        dataloader = generate_random_dataloader(
            num_samples=12, input_dim=2, batch_size=2
        )
        train(my_unit, dataloader, max_epochs=2, callbacks=[callback])
        # 2 epochs, 6 iterations each, logging every third step
        self.assertEqual(logger.add_scalar.call_count, 4)

    def test_warmup_steps(self) -> None:
        logger = MagicMock(spec=MetricLogger)
        callback = IterationTimeLogger(logger=logger, warmup_steps=1)
        timer = Timer()
        timer.recorded_durations = {"train_iteration_time": [1, 2]}

        # ensure that we don't log for the first step
        callback._log_step_metrics("train_iteration_time", timer, 1)
        logger.log.assert_not_called()

        # second step should log
        callback._log_step_metrics("train_iteration_time", timer, 2)
        self.assertEqual(logger.log.call_count, 1)

    def test_invalid_params(self) -> None:
        logger = MagicMock(spec=MetricLogger)
        with self.assertRaisesRegex(
            ValueError, "moving_avg_window must be at least 1, got 0"
        ):
            IterationTimeLogger(logger=logger, moving_avg_window=0)

        with self.assertRaisesRegex(
            ValueError, "log_every_n_steps must be at least 1, got -1"
        ):
            IterationTimeLogger(logger=logger, log_every_n_steps=-1)

        with self.assertRaisesRegex(
            ValueError, "warmup_steps must be at least 0, got -1"
        ):
            IterationTimeLogger(logger=logger, warmup_steps=-1)
