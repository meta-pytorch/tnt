#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock

from torchtnt.framework._test_utils import DummyTrainUnit, generate_random_dataloader
from torchtnt.framework.callbacks.learning_rate_monitor import LearningRateMonitor
from torchtnt.framework.train import train
from torchtnt.utils.loggers.logger import MetricLogger


class LearningRateMonitorTest(unittest.TestCase):
    def test_learning_rate_monitor_epoch(self) -> None:
        """
        Test LearningRateMonitor callback with 'epoch' logging interval
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        log_writer = MagicMock(spec=MetricLogger)
        monitor = LearningRateMonitor(loggers=log_writer)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[monitor])
        self.assertEqual(log_writer.log_dict.call_count, 2)

    def test_learning_rate_monitor_step(self) -> None:
        """
        Test LearningRateMonitor callback with 'step' logging interval
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        log_writer = MagicMock(spec=MetricLogger)
        monitor = LearningRateMonitor(loggers=log_writer, logging_interval="step")

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        total_steps = (dataset_len / batch_size) * max_epochs

        train(
            my_unit,
            dataloader,
            max_epochs=max_epochs,
            # pyre-fixme[6]: For 4th argument expected `Optional[int]` but got `float`.
            max_steps=total_steps,
            callbacks=[monitor],
        )
        self.assertEqual(log_writer.log_dict.call_count, total_steps)
