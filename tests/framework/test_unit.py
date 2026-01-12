#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Iterator

import torch
from torchtnt.framework._test_utils import get_dummy_train_state
from torchtnt.framework.state import State
from torchtnt.framework.unit import EvalUnit, PredictUnit, TrainUnit


class TestUnit(
    EvalUnit[Iterator[torch.Tensor]], PredictUnit[torch.Tensor], TrainUnit[torch.Tensor]
):
    def __init__(self) -> None:
        super().__init__()

    def train_step(self, state: State, data: torch.Tensor) -> None:
        return

    def eval_step(self, state: State, data: Iterator[torch.Tensor]) -> None:
        return

    def predict_step(self, state: State, data: torch.Tensor) -> None:
        return


class UnitTest(unittest.TestCase):
    def test_initialization_and_get_next_batch(self) -> None:
        unit = TestUnit()
        self.assertIsNotNone(unit.train_progress)
        self.assertIsNotNone(unit.eval_progress)
        self.assertIsNotNone(unit.predict_progress)

        tensor_1 = torch.ones(1)
        tensor_2 = torch.zeros(1)
        state = get_dummy_train_state()

        # test train next batch - exepct to return the elements within the iterable
        train_data_iter = iter([tensor_1, tensor_2])
        self.assertEqual(unit.get_next_train_batch(state, train_data_iter), tensor_1)
        self.assertEqual(unit.get_next_train_batch(state, train_data_iter), tensor_2)

        # test predict next batch - exepct to return the elements within the iterable
        self.assertEqual(
            unit.get_next_predict_batch(state, iter([tensor_1, tensor_2])), tensor_1
        )

        # test eval next batch - exepct to return the iterable
        data_iter = iter([tensor_1, tensor_2])
        next_eval_batch = unit.get_next_eval_batch(state, data_iter)
        self.assertEqual(next_eval_batch, data_iter)
