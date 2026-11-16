#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import time
import unittest
from unittest.mock import MagicMock, patch

import torch
from torchtnt.utils.test_utils import skip_if_not_gpu
from torchtnt.utils.timer import Timer


class TimerGPUTest(unittest.TestCase):
    @skip_if_not_gpu
    @patch("torch.cuda.synchronize")
    def test_timer_synchronize(self, mock_synchronize: MagicMock) -> None:
        """Make sure that torch.cuda.synchronize() is not called by default when GPU is present."""

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        timer = Timer()

        # Do not explicitly call synchronize, timer must call it for test to pass.

        with timer.time("action_1"):
            start_event.record()
            time.sleep(0.5)
            end_event.record()

        self.assertEqual(mock_synchronize.call_count, 0)

    @skip_if_not_gpu
    @patch("torch.cuda.synchronize")
    def test_timer_synchronize_when_explicit(self, mock_synchronize: MagicMock) -> None:
        """Make sure that torch.cuda.synchronize() is called when GPU is present and sync is explicit."""

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        timer = Timer(cuda_sync=True)

        # Do not explicitly call synchronize, timer must call it for test to pass.

        with timer.time("action_1"):
            start_event.record()
            time.sleep(0.5)
            end_event.record()

        self.assertEqual(mock_synchronize.call_count, 2)
