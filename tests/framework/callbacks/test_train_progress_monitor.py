#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from torchtnt.framework._test_utils import DummyTrainUnit, generate_random_dataloader
from torchtnt.framework.callbacks.train_progress_monitor import TrainProgressMonitor
from torchtnt.framework.train import train
from torchtnt.utils.loggers import InMemoryLogger


class TrainProgressMonitorTest(unittest.TestCase):
    def test_train_progress_monitor(self) -> None:
        """
        Test TrainProgressMonitor callback
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 3
        num_train_steps_per_epoch = dataset_len / batch_size

        my_unit = DummyTrainUnit(input_dim=input_dim)
        logger = InMemoryLogger()
        monitor = TrainProgressMonitor(loggers=logger)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[monitor])

        buf = logger.log_buffer
        self.assertEqual(
            len(buf), max_epochs + 1
        )  # +1 since we also log on_train_start
        self.assertEqual(
            buf[0]["Training steps completed vs epochs"], num_train_steps_per_epoch * 0
        )
        self.assertEqual(
            buf[1]["Training steps completed vs epochs"], num_train_steps_per_epoch * 1
        )
        self.assertEqual(
            buf[2]["Training steps completed vs epochs"], num_train_steps_per_epoch * 2
        )
        self.assertEqual(
            buf[3]["Training steps completed vs epochs"], num_train_steps_per_epoch * 3
        )
