#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock

import torch
from torchtnt.framework._test_utils import (
    DummyAutoUnit,
    DummyTrainUnit,
    generate_random_dataloader,
)
from torchtnt.framework.callbacks.module_summary import ModuleSummary
from torchtnt.framework.state import EntryPoint, PhaseState, State


class ModuleSummaryTest(unittest.TestCase):
    def test_module_summary_max_depth(self) -> None:
        """
        Test ModuleSummary callback with train entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 1

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = State(
            entry_point=EntryPoint.TRAIN,
            train_state=PhaseState(
                dataloader=dataloader,
                max_epochs=max_epochs,
            ),
        )

        my_unit = MagicMock(spec=DummyTrainUnit)
        module_summary_callback = ModuleSummary(max_depth=2)
        module_summary_callback.on_train_epoch_start(state, my_unit)
        self.assertEqual(module_summary_callback._max_depth, 2)

    def test_module_summary_retrieve_module_summaries(self) -> None:
        """
        Test ModuleSummary callback in train
        """

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l1 = torch.nn.Linear(2, 2)
                self.b1 = torch.nn.BatchNorm1d(2)
                self.l2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                x = self.l1(x)
                x = self.b1(x)
                x = self.l2(x)
                return x

        my_module = Net()

        auto_unit = DummyAutoUnit(
            module=my_module,
        )

        module_summary_callback = ModuleSummary()
        summaries = module_summary_callback._retrieve_module_summaries(auto_unit)

        self.assertEqual(len(summaries), 1)
        ms = summaries[0]
        self.assertEqual(ms.module_name, "module")
        self.assertEqual(ms.module_type, "Net")
        self.assertTrue("l1" in ms.submodule_summaries)
        self.assertTrue("b1" in ms.submodule_summaries)
        self.assertTrue("l2" in ms.submodule_summaries)

    def test_module_summary_retrieve_module_summaries_module_inputs(self) -> None:
        """
        Test ModuleSummary callback in train
        """

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l1 = torch.nn.Linear(2, 2)
                self.b1 = torch.nn.BatchNorm1d(2)
                self.l2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                x = self.l1(x)
                x = self.b1(x)
                x = self.l2(x)
                return x

        my_module = Net()

        auto_unit = DummyAutoUnit(
            module=my_module,
        )

        module_inputs = {"module": ((torch.rand(2, 2, device=auto_unit.device),), {})}
        module_summary_callback = ModuleSummary(module_inputs=module_inputs)
        summaries = module_summary_callback._retrieve_module_summaries(auto_unit)

        self.assertEqual(len(summaries), 1)
        ms = summaries[0]
        self.assertEqual(ms.flops_forward, 16)
        self.assertEqual(ms.flops_backward, 24)
        self.assertEqual(ms.in_size, [2, 2])
        self.assertTrue(ms.forward_elapsed_time_ms != "?")
