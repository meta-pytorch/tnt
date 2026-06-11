#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict, Iterator

import torch
from torch.optim import Optimizer
from torchtnt.framework._unit_utils import (
    _find_optimizers_for_module,
    _step_requires_iterator,
)
from torchtnt.framework.state import State


class UnitUtilsTest(unittest.TestCase):
    def test_step_func_requires_iterator(self) -> None:
        class Foo:
            def bar(self, state: State, data: object) -> object:
                return data

            def baz(self, state: State, data: Iterator[torch.Tensor]) -> object:
                pass

        def dummy(a: int, b: str, data: Iterator[str]) -> None:
            pass

        foo = Foo()

        self.assertFalse(_step_requires_iterator(foo.bar))
        self.assertTrue(_step_requires_iterator(foo.baz))
        # pyrefly: ignore [bad-argument-type]
        self.assertTrue(_step_requires_iterator(dummy))

    def test_find_optimizers_for_module(self) -> None:
        module1 = torch.nn.Linear(10, 10)
        module2 = torch.nn.Linear(10, 10)
        optim1 = torch.optim.Adam(module1.parameters())
        optim2 = torch.optim.Adagrad(module2.parameters())

        opts: Dict[str, Optimizer] = {"optim1": optim1, "optim2": optim2}
        optimizers = _find_optimizers_for_module(module1, opts)
        optim_name, _ = optimizers[0]
        self.assertEqual(optim_name, "optim1")
        optimizers = _find_optimizers_for_module(module2, opts)
        optim_name, _ = optimizers[0]
        self.assertEqual(optim_name, "optim2")
