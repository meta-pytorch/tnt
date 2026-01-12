#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import time
import unittest
from unittest.mock import MagicMock, patch

from torchtnt.framework.utils import get_timing_context
from torchtnt.utils.timer import Timer


class UtilsTest(unittest.TestCase):
    @patch("torchtnt.framework.utils.record_function")
    def test_get_timing_context(self, mock_record_function: MagicMock) -> None:
        state = MagicMock()
        state.timer = None

        ctx = get_timing_context(state, "a")
        with ctx:
            time.sleep(1)
        mock_record_function.assert_called_with("a")

        state.timer = Timer()
        ctx = get_timing_context(state, "b")
        with ctx:
            time.sleep(1)
        self.assertTrue("b" in state.timer.recorded_durations.keys())
        mock_record_function.assert_called_with("b")
