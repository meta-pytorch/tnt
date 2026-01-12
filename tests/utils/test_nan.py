#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torchtnt.utils.nan import check_for_nan_or_inf, register_nan_hooks_on_whole_graph


class NaNFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore overrides method defined in `torch.autograd.function._SingleLevelFunction` inconsistently
    def forward(ctx, input):
        return input.clone()

    @staticmethod
    # pyre-ignore overrides method defined in `torch.autograd.function._SingleLevelFunction` inconsistently
    def backward(ctx, grad_output):
        return torch.tensor([float("nan")], device="cpu")


class NanHookTest(unittest.TestCase):
    def test_register_nan_hooks_on_whole_graph(self) -> None:
        x = torch.tensor([1.0], device="cpu", requires_grad=True)
        out = NaNFunction.apply(x)

        # no error is thrown
        out.backward()

        _ = register_nan_hooks_on_whole_graph([out])
        with self.assertRaisesRegex(RuntimeError, "Detected NaN"):
            out.backward()

    def test_check_for_nan_or_inf(self) -> None:
        tensor = torch.tensor([float("nan")], device="cpu")

        with self.assertRaisesRegex(RuntimeError, "Detected NaN or Inf in tensor"):
            check_for_nan_or_inf(tensor)

        tensor = torch.tensor([float("inf")], device="cpu")
        with self.assertRaisesRegex(RuntimeError, "Detected NaN or Inf in tensor"):
            check_for_nan_or_inf(tensor)
