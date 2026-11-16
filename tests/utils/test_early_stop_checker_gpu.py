#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torchtnt.utils.early_stop_checker import EarlyStopChecker
from torchtnt.utils.test_utils import skip_if_not_gpu


class EarlyStopCheckerGPUTest(unittest.TestCase):
    @skip_if_not_gpu
    def test_early_stop_min_delta_on_gpu(self) -> None:
        device = torch.device("cuda:0")

        # Loss decreases beyond 0.25 but not more than min_delta
        losses = [
            torch.tensor([0.4], device=device),
            torch.tensor([0.38], device=device),
            torch.tensor([0.31], device=device),
            torch.tensor([0.25], device=device),
            torch.tensor([0.27], device=device),
            torch.tensor([0.24], device=device),
        ]
        es1 = EarlyStopChecker("min", 3, min_delta=0.05)
        es2 = EarlyStopChecker("min", 4, min_delta=0.05)

        for loss in losses:
            should_stop = es1.check(torch.tensor(loss))
            self.assertFalse(should_stop)
            should_stop = es2.check(torch.tensor(loss))
            self.assertFalse(should_stop)

        # Patience should run out
        should_stop = es1.check(torch.tensor(0.25))
        self.assertTrue(should_stop)

        # es2 has more patience than es1
        should_stop = es2.check(torch.tensor(0.25))
        self.assertFalse(should_stop)
        should_stop = es2.check(torch.tensor(0.26))
        self.assertTrue(should_stop)
