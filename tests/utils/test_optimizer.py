#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torchtnt.utils.env import init_from_env
from torchtnt.utils.optimizer import init_optim_state


class OptimizerTest(unittest.TestCase):
    def test_init_optim_state(self) -> None:
        """Test optimizer skeleton state initialization."""
        device = init_from_env()
        module = torch.nn.Linear(1, 1, device=device)
        original_state_dict = module.state_dict().copy()
        optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
        self.assertEqual(optimizer.state, {})

        init_optim_state(optimizer)

        # check that optimizer state has been initialized
        self.assertNotEqual(optimizer.state, {})

        # check that parameters have not changed
        self.assertTrue(
            torch.allclose(
                original_state_dict["weight"],
                module.state_dict()["weight"],
            )
        )
        self.assertTrue(
            torch.allclose(
                original_state_dict["bias"],
                module.state_dict()["bias"],
            )
        )
