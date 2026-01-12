#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import ANY, call, MagicMock, patch

import torch
from pyre_extensions import none_throws
from torchtnt.framework._callback_handler import CallbackHandler
from torchtnt.framework._test_utils import (
    DummyAutoUnit,
    DummyPredictUnit,
    DummyTrainUnit,
    generate_random_dataloader,
)
from torchtnt.framework.callbacks.throughput_logger import ThroughputLogger
from torchtnt.framework.predict import predict
from torchtnt.framework.state import ActivePhase, EntryPoint, PhaseState, State
from torchtnt.framework.train import _train_impl, train
from torchtnt.framework.unit import TrainUnit
from torchtnt.utils.loggers.logger import MetricLogger


class ThroughputLoggerTest(unittest.TestCase):
    def test_maybe_log_for_step(self) -> None:
        logger = MagicMock(spec=MetricLogger)
        throughput_logger = ThroughputLogger(logger, {"Batches": 1, "Items": 32})
        phase_state = PhaseState(dataloader=[])
        phase_state.iteration_timer.recorded_durations = {
            "data_wait_time": [1, 4],
            "train_iteration_time": [3],
        }
        state = State(entry_point=EntryPoint.TRAIN, train_state=phase_state)
        throughput_logger._maybe_log_for_step(state, 1)
        logger.log.assert_has_calls(
            [
                call(
                    "Train: Batches per second (step granularity)",
                    0.25,  # 1/(1+3)
                    1,
                ),
                call(
                    "Train: Items per second (step granularity)",
                    8,  # 32/(1+3)
                    1,
                ),
            ],
            any_order=True,
        )
        logger.log.reset_mock()
        phase_state.iteration_timer.recorded_durations["train_iteration_time"].append(4)
        throughput_logger._maybe_log_for_step(state, 2, is_step_end_hook=False)
        logger.log.assert_has_calls(
            [
                call(
                    "Train: Batches per second (step granularity)",
                    0.125,  # 1/(4+4)
                    2,
                ),
                call(
                    "Train: Items per second (step granularity)",
                    4,  # 32/(4+4)
                    2,
                ),
            ]
        )

    def test_maybe_log_for_step_early_return(self) -> None:
        logger = MagicMock(spec=MetricLogger)
        throughput_logger = ThroughputLogger(logger, {"Batches": 1})
        phase_state = PhaseState(dataloader=[])
        recorded_durations_dict = {
            "data_wait_time": [0.0, 4.0],
            "train_iteration_time": [0.0],
        }
        # total_time <= 0
        phase_state.iteration_timer.recorded_durations = recorded_durations_dict
        state = State(entry_point=EntryPoint.TRAIN, train_state=phase_state)
        throughput_logger._maybe_log_for_step(state, step_logging_for=1)
        logger.log.assert_not_called()

        # empty iteration_time_list
        recorded_durations_dict["data_wait_time"] = [1.0, 2.0]
        recorded_durations_dict["train_iteration_time"] = []
        throughput_logger._maybe_log_for_step(state, step_logging_for=1)
        logger.log.assert_not_called()

        # small data_wait_time list
        recorded_durations_dict["data_wait_time"] = [1.0]
        recorded_durations_dict["train_iteration_time"] = [1.0]
        throughput_logger._maybe_log_for_step(state, step_logging_for=1)
        logger.log.assert_not_called()

        # step_logging_for % log_every_n_steps != 0
        recorded_durations_dict["data_wait_time"] = [1.0, 2.0]
        throughput_logger = ThroughputLogger(
            logger, {"Batches": 1}, log_every_n_steps=2
        )
        throughput_logger._maybe_log_for_step(state, step_logging_for=1)
        logger.log.assert_not_called()

    def test_with_comparing_time(self) -> None:
        logger = MagicMock(spec=MetricLogger)
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
                evaluate_every_n_epochs=2,
            ),
        )
        throughput_logger = ThroughputLogger(
            logger=logger,
            throughput_per_batch={"Batches": 1, "Queries": 8},
            log_every_n_steps=1,
        )

        # we want to be able to compare the logging value to the state, so we need to create state manually and
        # call _train_impl. This would have been similar to calling fit() and getting the state as a ret value
        _train_impl(
            state,
            DummyAutoUnit(module=torch.nn.Linear(2, 2)),
            CallbackHandler([throughput_logger]),
        )

        train_iteration_times = none_throws(
            state.train_state
        ).iteration_timer.recorded_durations["train_iteration_time"]
        train_twfb_times = none_throws(
            state.train_state
        ).iteration_timer.recorded_durations["data_wait_time"]
        eval_iteration_times = none_throws(
            state.eval_state
        ).iteration_timer.recorded_durations["eval_iteration_time"]
        eval_twfb_times = none_throws(
            state.eval_state
        ).iteration_timer.recorded_durations["data_wait_time"]

        self.assertEqual(len(train_iteration_times), 4)
        self.assertEqual(len(train_twfb_times), 4)
        self.assertEqual(len(eval_iteration_times), 2)
        self.assertEqual(len(eval_twfb_times), 2)

        train_step_times = [
            train_iteration_times[i] + train_twfb_times[i] for i in range(4)
        ]
        eval_step_times = [
            eval_iteration_times[i] + eval_twfb_times[i] for i in range(2)
        ]
        self.assertEqual(
            logger.log.call_count, 18
        )  # steps: 8 train (2epochs x 2steps x 2items), 4 eval (1x2x2). epochs: 4 train (2epoch x 2items). 2 eval (1x2)
        train_batches_step_logs = [
            call(
                "Train: Batches per second (step granularity)",
                1 / (train_step_times[i]),
                i + 1,
            )
            for i in range(4)
        ]
        train_queries_step_logs = [
            call(
                "Train: Queries per second (step granularity)",
                8 / (train_step_times[i]),
                i + 1,
            )
            for i in range(4)
        ]
        eval_batches_step_logs = [
            call(
                "Eval: Batches per second (step granularity)",
                1 / (eval_step_times[i]),
                i + 1,
            )
            for i in range(2)
        ]
        eval_queries_step_logs = [
            call(
                "Eval: Queries per second (step granularity)",
                8 / (eval_step_times[i]),
                i + 1,
            )
            for i in range(2)
        ]
        # for epoch, we test the logged value separately
        train_batches_epoch_logs = [
            call("Train: Batches per second (epoch granularity)", ANY, i)
            for i in range(1, 3)
        ]
        train_queries_epoch_logs = [
            call("Train: Queries per second (epoch granularity)", ANY, i)
            for i in range(1, 3)
        ]
        eval_epoch_logs = [
            call(
                "Eval: Queries per second (epoch granularity)",
                ANY,
                1,
            ),
            call(
                "Eval: Batches per second (epoch granularity)",
                ANY,
                1,
            ),
        ]

        logger.log.assert_has_calls(
            train_batches_step_logs
            + train_queries_step_logs
            + eval_batches_step_logs
            + eval_queries_step_logs
            + train_batches_epoch_logs
            + train_queries_epoch_logs
            + eval_epoch_logs,
            any_order=True,
        )

    def test_with_predict(self) -> None:
        logger = MagicMock(spec=MetricLogger)
        predict(
            DummyPredictUnit(input_dim=2),
            generate_random_dataloader(num_samples=8, input_dim=2, batch_size=2),
            max_steps_per_epoch=1,
            callbacks=[
                ThroughputLogger(
                    logger=logger,
                    throughput_per_batch={"Batches": 1},
                    log_every_n_steps=1,
                )
            ],
        )
        logger.log.assert_has_calls(
            [
                call(
                    "Predict: Batches per second (step granularity)",
                    ANY,
                    1,
                )
            ],
            [
                call(
                    "Predict: Batches per second (epoch granularity)",
                    ANY,
                    1,
                )
            ],
        )

    def test_log_for_epoch(self) -> None:
        logger = MagicMock(spec=MetricLogger)
        unit = DummyTrainUnit(input_dim=2)
        throughput_logger = ThroughputLogger(logger, {"Batches": 1, "Queries": 8})
        state = State(entry_point=EntryPoint.TRAIN)

        self.assertIsNone(throughput_logger._epoch_start_times.get(ActivePhase.TRAIN))
        self.assertEqual(throughput_logger._steps_in_epoch[ActivePhase.TRAIN], 0)
        with patch.object(throughput_logger, "_maybe_log_for_step"):
            throughput_logger.on_train_step_end(state, unit)
        self.assertEqual(throughput_logger._steps_in_epoch[ActivePhase.TRAIN], 1)

        # Make sure we don't log or fail if the _epoch_start_times dict is not initialized
        throughput_logger._log_for_epoch(state, epoch_logging_for=15)
        logger.log.assert_not_called()

        with patch("time.perf_counter", return_value=0.5):
            throughput_logger.on_train_epoch_start(state, MagicMock(spec=TrainUnit))
        self.assertEqual(throughput_logger._epoch_start_times[ActivePhase.TRAIN], 0.5)

        throughput_logger._steps_in_epoch[ActivePhase.TRAIN] = (
            2  # to assume there were two steps in the epoch
        )
        logger.log.reset_mock()
        with patch("time.perf_counter", return_value=0.6):
            throughput_logger._log_for_epoch(state, epoch_logging_for=15)

        logger.log.assert_has_calls(
            [
                call(
                    "Train: Batches per second (epoch granularity)",
                    (1 * 2) / (0.6 - 0.5),
                    15,
                ),
                call(
                    "Train: Queries per second (epoch granularity)",
                    (8 * 2) / (0.6 - 0.5),
                    15,
                ),
            ]
        )

    def test_epoch_logging_time(self) -> None:
        logger = MagicMock(spec=MetricLogger)
        throughput_logger = ThroughputLogger(logger, {"Queries": 4})
        with patch("time.perf_counter", side_effect=[0.1, 0.5, 0.8, 1.5]):
            train(
                DummyTrainUnit(input_dim=2),
                generate_random_dataloader(num_samples=16, input_dim=2, batch_size=4),
                max_epochs=2,
                max_steps_per_epoch=2,
                callbacks=[throughput_logger],
            )

        logger.log.assert_has_calls(
            [
                call(
                    "Train: Queries per second (epoch granularity)",
                    (4 * 2) / (0.5 - 0.1),
                    1,
                ),
                call(
                    "Train: Queries per second (epoch granularity)",
                    (4 * 2) / (1.5 - 0.8),
                    2,
                ),
            ],
            any_order=True,
        )

    def test_warmup_steps(self) -> None:
        logger = MagicMock(spec=MetricLogger)
        throughput_logger = ThroughputLogger(
            logger, {"Batches": 1, "Items": 32}, warmup_steps=1
        )
        phase_state = PhaseState(dataloader=[])
        phase_state.iteration_timer.recorded_durations = {
            "data_wait_time": [1, 4],
            "train_iteration_time": [3],
        }
        state = State(entry_point=EntryPoint.TRAIN, train_state=phase_state)

        throughput_logger._maybe_log_for_step(state, 1)
        logger.log.assert_not_called()

        throughput_logger._maybe_log_for_step(state, 2)
        self.assertEqual(logger.log.call_count, 2)

    def test_input_validation(self) -> None:
        logger = MagicMock(spec=MetricLogger)
        with self.assertRaisesRegex(ValueError, "throughput_per_batch cannot be empty"):
            ThroughputLogger(logger, {})

        with self.assertRaisesRegex(
            ValueError, "throughput_per_batch item Batches must be at least 1, got -1"
        ):
            ThroughputLogger(logger, {"Queries": 8, "Batches": -1})

        with self.assertRaisesRegex(
            ValueError, "log_every_n_steps must be at least 1, got 0"
        ):
            ThroughputLogger(logger, {"Batches": 1}, log_every_n_steps=0)

        with self.assertRaisesRegex(
            ValueError, "warmup_steps must be at least 0, got -1"
        ):
            ThroughputLogger(logger, {"Batches": 1}, warmup_steps=-1)
