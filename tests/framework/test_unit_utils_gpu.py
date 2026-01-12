#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer
from torchtnt.framework._unit_utils import _find_optimizers_for_module
from torchtnt.utils.distributed import spawn_multi_process
from torchtnt.utils.env import init_from_env
from torchtnt.utils.test_utils import skip_if_not_distributed, skip_if_not_gpu


class UnitUtilsGPUTest(unittest.TestCase):
    @skip_if_not_distributed
    @skip_if_not_gpu
    def test_find_optimizers_for_FSDP_module(self) -> None:
        spawn_multi_process(2, "nccl", self._find_optimizers_for_FSDP_module)

    @staticmethod
    def _find_optimizers_for_FSDP_module() -> None:
        device = init_from_env()
        module1 = FSDP(torch.nn.Linear(10, 10).to(device))
        module2 = torch.nn.Linear(10, 10)
        optim1 = torch.optim.Adam(module1.parameters())
        optim2 = torch.optim.Adagrad(module2.parameters())

        opts: Dict[str, Optimizer] = {"optim1": optim1, "optim2": optim2}
        optim_list = _find_optimizers_for_module(module1, opts)
        optim_name, _ = optim_list[0]

        tc = unittest.TestCase()
        tc.assertEqual(optim_name, "optim1")
        optim_list = _find_optimizers_for_module(module2, opts)
        optim_name, _ = optim_list[0]
        tc.assertEqual(optim_name, "optim2")
