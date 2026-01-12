#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from copy import deepcopy

import torch
from torchtnt.utils.misc import (
    days_to_secs,
    transfer_batch_norm_stats,
    transfer_weights,
)


class MiscTest(unittest.TestCase):
    def test_days_to_secs(self) -> None:
        self.assertIsNone(days_to_secs(None))
        self.assertEqual(days_to_secs(1), 60 * 60 * 24)
        with self.assertRaises(ValueError):
            days_to_secs(-1)

    def test_transfer_weights(self) -> None:
        module1 = torch.nn.Linear(2, 2)
        module2 = torch.nn.Linear(2, 2)
        module3 = deepcopy(module2)
        self.assertFalse(
            torch.allclose(
                module1.state_dict()["weight"], module2.state_dict()["weight"]
            )
        )
        self.assertFalse(
            torch.allclose(
                module1.state_dict()["weight"], module3.state_dict()["weight"]
            )
        )
        self.assertTrue(
            torch.allclose(
                module2.state_dict()["weight"], module3.state_dict()["weight"]
            )
        )
        transfer_weights(module1, module2)
        self.assertTrue(
            torch.allclose(
                module1.state_dict()["weight"], module2.state_dict()["weight"]
            )
        )
        self.assertFalse(
            torch.allclose(
                module2.state_dict()["weight"], module3.state_dict()["weight"]
            )
        )

    def test_transfer_batch_norm_stats(self) -> None:
        module1 = torch.nn.BatchNorm2d(3)
        # change running mean and var
        # pyre-fixme[8]: Attribute has type `Optional[Tensor]`; used as `int`.
        # pyre-fixme[58]: `+` is not supported for operand types
        #  `Optional[torch._tensor.Tensor]` and `int`.
        module1.running_mean = module1.running_mean + 2
        # pyre-fixme[8]: Attribute has type `Optional[Tensor]`; used as `int`.
        # pyre-fixme[58]: `+` is not supported for operand types
        #  `Optional[torch._tensor.Tensor]` and `int`.
        module1.running_var = module1.running_var + 4
        module2 = torch.nn.BatchNorm2d(3)

        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Optional[Tensor]`.
        # pyre-fixme[6]: For 2nd argument expected `Tensor` but got `Optional[Tensor]`.
        self.assertFalse(torch.equal(module1.running_mean, module2.running_mean))
        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Optional[Tensor]`.
        # pyre-fixme[6]: For 2nd argument expected `Tensor` but got `Optional[Tensor]`.
        self.assertFalse(torch.equal(module1.running_var, module2.running_var))
        transfer_batch_norm_stats(module1, module2)
        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Optional[Tensor]`.
        # pyre-fixme[6]: For 2nd argument expected `Tensor` but got `Optional[Tensor]`.
        self.assertTrue(torch.equal(module1.running_mean, module2.running_mean))
        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Optional[Tensor]`.
        # pyre-fixme[6]: For 2nd argument expected `Tensor` but got `Optional[Tensor]`.
        self.assertTrue(torch.equal(module1.running_var, module2.running_var))
