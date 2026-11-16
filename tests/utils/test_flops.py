#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torchtnt.utils.flops import FlopTensorDispatchMode


class ModuleSummaryTest(unittest.TestCase):
    def test_torch_operations(self) -> None:
        """Make sure FLOPs calculation works for a single operations."""

        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                bmm_mat = torch.randn(10, 5, 7)
                mm_mat = torch.randn(7, 3)
                return x.bmm(bmm_mat).matmul(mm_mat)

        test_module = TestModule()
        with FlopTensorDispatchMode(test_module) as ftdm:
            inp = torch.randn(10, 4, 5)
            res = test_module(inp)

            self.assertEqual(res.shape[0], 10)
            self.assertEqual(res.shape[1], 4)
            self.assertEqual(res.shape[2], 3)

            self.assertEqual(
                ftdm.flop_counts[""].get("bmm.default", 0)
                + ftdm.flop_counts[""].get("bmm", 0),
                1400,
            )
            self.assertEqual(
                ftdm.flop_counts[""].get("mm.default", 0)
                + ftdm.flop_counts[""].get("mm", 0),
                840,
            )

            inp = torch.randn(10, 4, 5)
            inp = torch.autograd.Variable(inp, requires_grad=True)

            ftdm.reset()
            res = test_module(inp)
            res.mean().backward()

            self.assertEqual(res.shape[0], 10)
            self.assertEqual(res.shape[1], 4)
            self.assertEqual(res.shape[2], 3)

            self.assertEqual(
                ftdm.flop_counts[""].get("bmm.default", 0)
                + ftdm.flop_counts[""].get("bmm", 0),
                2800,
            )
            self.assertEqual(
                ftdm.flop_counts[""].get("mm.default", 0)
                + ftdm.flop_counts[""].get("mm", 0),
                1680,
            )

    def test_torch_linear_layer(self) -> None:
        """Make sure FLOPs calculation works for a module consists of linear layers."""
        lnn = torch.nn.Sequential(
            torch.nn.Sequential(torch.nn.Linear(10, 70), torch.nn.Linear(70, 5)),
            torch.nn.Linear(5, 1),
        )
        inp = torch.randn(1, 10)

        with FlopTensorDispatchMode(lnn) as ftdm:
            self.assertEqual(len(ftdm._all_hooks), 8)

            res = lnn(inp)
            self.assertEqual(
                ftdm.flop_counts[""].get("addmm.default", 0)
                + ftdm.flop_counts[""].get("addmm", 0),
                1055,
            )
            self.assertEqual(
                ftdm.flop_counts["0"].get("addmm.default", 0)
                + ftdm.flop_counts["0"].get("addmm", 0),
                1050,
            )
            self.assertEqual(
                ftdm.flop_counts["0.0"].get("addmm.default", 0)
                + ftdm.flop_counts["0.0"].get("addmm", 0),
                700,
            )
            self.assertEqual(
                ftdm.flop_counts["0.1"].get("addmm.default", 0)
                + ftdm.flop_counts["0.1"].get("addmm", 0),
                350,
            )
            self.assertEqual(
                ftdm.flop_counts["1"].get("addmm.default", 0)
                + ftdm.flop_counts["1"].get("addmm", 0),
                5,
            )
            ftdm.reset()
            res.backward()
            self.assertEqual(
                ftdm.flop_counts[""].get("mm.default", 0)
                + ftdm.flop_counts[""].get("mm", 0),
                1410,
            )
            self.assertEqual(
                ftdm.flop_counts["0"].get("mm.default", 0)
                + ftdm.flop_counts["0"].get("mm", 0),
                1400,
            )
            self.assertEqual(
                ftdm.flop_counts["0.0"].get("mm.default", 0)
                + ftdm.flop_counts["0.0"].get("mm", 0),
                700,
            )
            self.assertEqual(
                ftdm.flop_counts["0.1"].get("mm.default", 0)
                + ftdm.flop_counts["0.1"].get("mm", 0),
                700,
            )
            self.assertEqual(
                ftdm.flop_counts["1"].get("mm.default", 0)
                + ftdm.flop_counts["1"].get("mm", 0),
                10,
            )
