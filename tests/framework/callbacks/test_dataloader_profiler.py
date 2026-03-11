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
from torchtnt.framework._test_utils import DummyTrainUnit, generate_random_dataloader
from torchtnt.framework.callbacks.dataloader_profiler import (
    _empty_train_step,
    DataloaderProfiler,
)

from torchtnt.framework.state import State
from torchtnt.framework.train import train
from torchtnt.utils.loggers.logger import MetricLogger


class DataloaderProfilerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger_mock = MagicMock(spec=MetricLogger)
        self.logger_mock.log = MagicMock()

    def test_dataloader_only_mode_train_step_is_overriden(self) -> None:
        dl_profiler = DataloaderProfiler(
            mode=DataloaderProfiler.DATALOADER_ONLY,
            batch_size=2,
            logger=self.logger_mock,
            num_steps_to_profile=3,
        )
        state = MagicMock(spec=State)
        unit = DummyTrainUnit(input_dim=2)
        dl_profiler.on_train_start(state, unit)
        self.assertEqual(unit.train_step, _empty_train_step)
        self.assertIsNone(
            unit.train_step(state, (torch.zeros(1), torch.zeros(1))), None
        )

    def test_with_train(self) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        dataloader = generate_random_dataloader(20, 2, 4)
        dl_profiler = DataloaderProfiler(
            mode=DataloaderProfiler.DATALOADER_ONLY,
            batch_size=4,
            logger=self.logger_mock,
            num_steps_to_profile=1,
        )
        train(my_unit, dataloader, callbacks=[dl_profiler])
        self.assertEqual(self.logger_mock.log.call_count, 3)
        self.logger_mock.log.assert_has_calls(
            [
                call(
                    "DL Profiler: QPS",
                    ANY,
                    1,
                ),
                call(
                    "DL Profiler: Time waiting for batch (seconds)",
                    ANY,
                    1,
                ),
                call(
                    "DL Profiler: Train iteration time (seconds)",
                    ANY,
                    1,
                ),
            ]
        )
        self.assertEqual(
            my_unit.train_progress.num_steps_completed, 1
        )  # train stops after one step

    def test_param_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalide mode: foo"):
            DataloaderProfiler(
                mode="foo",
                num_steps_to_profile=1,
                batch_size=1,
                logger=self.logger_mock,
            )

        with self.assertRaisesRegex(
            ValueError,
            "Expected num_steps_to_profile to be a positive integer, but got -1",
        ):
            DataloaderProfiler(
                mode=DataloaderProfiler.DATALOADER_ONLY,
                num_steps_to_profile=-1,
                batch_size=1,
                logger=self.logger_mock,
            )

        with self.assertRaisesRegex(
            ValueError,
            "Expected batch_size to be a positive integer, but got -1",
        ):
            DataloaderProfiler(
                mode=DataloaderProfiler.DATALOADER_ONLY,
                num_steps_to_profile=1,
                batch_size=-1,
                logger=self.logger_mock,
            )
