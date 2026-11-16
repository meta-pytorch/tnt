#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math
import unittest

from torchtnt.utils.anomaly_evaluation import IsNaNEvaluator, ThresholdEvaluator


class TestAnomalyLogger(unittest.TestCase):

    def test_threshold(self) -> None:
        threshold = ThresholdEvaluator(min_val=0.5, max_val=0.9)
        self.assertFalse(threshold.is_anomaly())

        threshold.update(0.4)
        self.assertTrue(threshold.is_anomaly())

        threshold.update(0.6)
        self.assertFalse(threshold.is_anomaly())

        threshold.update(0.95)
        self.assertTrue(threshold.is_anomaly())

        threshold = ThresholdEvaluator(max_val=1)

        threshold.update(100.0)
        self.assertTrue(threshold.is_anomaly())

        threshold.update(-500.0)
        self.assertFalse(threshold.is_anomaly())

    def test_isnan(self) -> None:
        isnan = IsNaNEvaluator()
        self.assertFalse(isnan.is_anomaly())

        isnan.update(0.4)
        self.assertFalse(isnan.is_anomaly())

        isnan.update(math.nan)
        self.assertTrue(isnan.is_anomaly())
