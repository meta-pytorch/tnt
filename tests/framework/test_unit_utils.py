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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer
from torchtnt.framework._unit_utils import (
    _find_optimizers_for_module,
    _step_requires_iterator,
)
from torchtnt.framework.state import State
from torchtnt.utils.distributed import spawn_multi_process
from torchtnt.utils.test_utils import skip_if_not_distributed


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

    def test_find_optimizers_for_module_with_multiple_param_groups(self) -> None:
        module = torch.nn.Linear(10, 10)
        optim = torch.optim.AdamW(  # noqa: CITRINE(missing_for_each_optimizer)
            [
                {"params": [module.weight]},
                {"params": [module.bias], "weight_decay": 0.0},
            ]
        )

        optimizers = _find_optimizers_for_module(module, {"optim": optim})
        self.assertEqual([name for name, _ in optimizers], ["optim"])

    def test_find_optimizers_for_module_skips_optimizer_missing_params(self) -> None:
        module = torch.nn.Linear(10, 10)
        optim = torch.optim.SGD([module.weight], lr=0.1)  # noqa: CITRINE(missing_for_each_optimizer)

        self.assertEqual(_find_optimizers_for_module(module, {"optim": optim}), [])

    @skip_if_not_distributed
    def test_find_optimizers_for_FSDP_module_with_uneven_param_groups(self) -> None:
        spawn_multi_process(
            4, "gloo", self._find_optimizers_for_FSDP_module_with_uneven_param_groups
        )

    @staticmethod
    def _find_optimizers_for_FSDP_module_with_uneven_param_groups() -> None:
        # The norm is last in the flat param, so its data is only in the last rank's
        # shard and is a size-0 tensor on every other rank.
        module = FSDP(
            torch.nn.Sequential(
                torch.nn.Linear(16, 16, bias=False),
                torch.nn.Linear(16, 16, bias=False),
                torch.nn.Linear(16, 16, bias=False),
                torch.nn.LayerNorm(16, bias=False),
            ),
            use_orig_params=True,
            device_id=torch.device("cpu"),
        )
        params = list(module.parameters())
        optim = torch.optim.AdamW(  # noqa: CITRINE(missing_for_each_optimizer)
            [{"params": params[:3]}, {"params": params[3:], "weight_decay": 0.0}]
        )

        optimizers = _find_optimizers_for_module(module, {"optim": optim})
        tc = unittest.TestCase()
        tc.assertEqual([name for name, _ in optimizers], ["optim"])
