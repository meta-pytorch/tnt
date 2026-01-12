#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import ANY, call, MagicMock

import torch
from pyre_extensions import none_throws
from torch.utils.tensorboard import SummaryWriter
from torchtnt.framework._callback_handler import CallbackHandler
from torchtnt.framework._test_utils import (
    DummyAutoUnit,
    DummyPredictUnit,
    generate_random_dataloader,
)
from torchtnt.framework.callbacks.time_wait_for_batch_logger import (
    TimeWaitForBatchLogger,
)
from torchtnt.framework.predict import predict
from torchtnt.framework.state import EntryPoint, PhaseState, State
from torchtnt.framework.train import _train_impl
from torchtnt.utils.loggers.logger import MetricLogger
from torchtnt.utils.timer import Timer, TimerProtocol


class TimeWaitForBatchLoggerTest(unittest.TestCase):
    def test_log_step_metrics(self) -> None:
        for spec in [MetricLogger, SummaryWriter]:
            with self.subTest(spec=spec):
                logger = MagicMock(spec=spec)
                log_method = logger.log if spec is MetricLogger else logger.add_scalar

                twfb_logger = TimeWaitForBatchLogger(logger=logger, log_every_n_steps=2)
                timer = MagicMock(spec=TimerProtocol)
                timer.recorded_durations = {"data_wait_time": [1, 2, 3]}
                twfb_logger._log_step_metrics(timer=timer, label="foo", step=1)
                log_method.assert_not_called()
                twfb_logger._log_step_metrics(timer=timer, label="foo", step=2)
                log_method.assert_has_calls(
                    [
                        call(
                            "foo",
                            3,  # last element in the data wait time list
                            2,  # step
                        )
                    ],
                )

    def test_comparing_twfb_logging_time(self) -> None:
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
                max_steps_per_epoch=1,
                evaluate_every_n_epochs=1,
            ),
        )

        logger = MagicMock(spec=MetricLogger)
        # we want to be able to compare the logging value to the state, so we need to create state manually and
        # call _train_impl. This would have been similar to calling fit() and getting the state as a ret value
        _train_impl(
            state,
            DummyAutoUnit(module=torch.nn.Linear(2, 2)),
            CallbackHandler(
                [TimeWaitForBatchLogger(logger=logger, log_every_n_steps=1)]
            ),
        )
        train_twfb_durations = none_throws(
            state.train_state
        ).iteration_timer.recorded_durations["data_wait_time"]
        eval_iteration_timer = none_throws(
            state.eval_state
        ).iteration_timer.recorded_durations["data_wait_time"]

        expected_training_iteration_time_calls = [
            call("Time Wait For Batch (Train)", train_twfb_durations[i], i + 1)
            for i in range(4)
        ]
        expected_eval_iteration_time_calls = [
            call("Time Wait For Batch (Eval)", eval_iteration_timer[i], i + 1)
            for i in range(2)
        ]

        logger.log.assert_has_calls(
            expected_training_iteration_time_calls + expected_eval_iteration_time_calls,
            any_order=True,
        )

    def test_with_predict(self) -> None:
        logger = MagicMock(spec=MetricLogger)
        predict(
            DummyPredictUnit(input_dim=2),
            generate_random_dataloader(num_samples=8, input_dim=2, batch_size=2),
            max_steps_per_epoch=1,
            callbacks=[TimeWaitForBatchLogger(logger=logger, log_every_n_steps=1)],
        )
        logger.log.assert_has_calls(
            [
                call(
                    "Time Wait For Batch (Predict)",
                    ANY,
                    1,
                )
            ],
        )

    def test_warmup_steps(self) -> None:
        logger = MagicMock(spec=MetricLogger)
        callback = TimeWaitForBatchLogger(logger=logger, warmup_steps=1)
        timer = Timer()
        timer.recorded_durations = {"data_wait_time": [1, 2]}

        # ensure that we don't log for the first step
        callback._log_step_metrics(timer=timer, label="foo", step=1)
        logger.log.assert_not_called()

        # second step should log
        callback._log_step_metrics(timer=timer, label="foo", step=2)
        self.assertEqual(logger.log.call_count, 1)

    def test_invalid_params(self) -> None:
        logger_mock = MagicMock(spec=MetricLogger)
        with self.assertRaisesRegex(
            ValueError, "log_every_n_steps must be at least 1, got 0"
        ):
            TimeWaitForBatchLogger(logger=logger_mock, log_every_n_steps=0)

        with self.assertRaisesRegex(
            ValueError, "warmup_steps must be at least 0, got -1"
        ):
            TimeWaitForBatchLogger(logger=logger_mock, warmup_steps=-1)
