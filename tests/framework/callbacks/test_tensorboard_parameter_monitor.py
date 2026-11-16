#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from unittest.mock import MagicMock

from torch.utils.tensorboard import SummaryWriter
from torchtnt.framework._test_utils import DummyTrainUnit, generate_random_dataloader
from torchtnt.framework.callbacks.tensorboard_parameter_monitor import (
    TensorBoardParameterMonitor,
)
from torchtnt.framework.train import train


class TensorBoardParameterMonitorTest(unittest.TestCase):
    def test_monitor_train(self) -> None:
        """
        Test TensorBoardParameterMonitor callback with train entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        summary_writer = MagicMock(spec=SummaryWriter)
        monitor = TensorBoardParameterMonitor(logger=summary_writer)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[monitor])
        # pyre-fixme[6]: For 2nd argument expected `SupportsDunderLT[Variable[_T]]`
        #  but got `int`.
        self.assertGreater(summary_writer.add_histogram.call_count, 0)
