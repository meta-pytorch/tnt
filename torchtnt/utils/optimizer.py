#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict

import torch


def init_optim_state(optimizer: torch.optim.Optimizer) -> None:
    """
    Initialize optimizer states by calling step() with zero grads. This is necessary because some optimizers like AdamW
    initialize some states in their state_dicts lazily, only after calling step() for the first time. Certain checkpointing
    solutions may rely on in-place loading, re-using existing tensor allocated memory from the optimizer state dict. This
    optimization does not work with optimizers that lazily initialize their states, as certain states will not be restored.
    Calling this function ensures that these states are available in the state dict for in place loading.

    Args:
        optimizer: A PyTorch optimizer.
    """
    if optimizer.state:
        # The optimizer state is initialized.
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if param.grad is not None:
                raise RuntimeError(
                    "Initializing the optimizer states requires that no existing gradients for parameters are found."
                )
            if param.requires_grad:
                param.grad = torch.zeros_like(param)
    optimizer.step(closure=None)
    optimizer.zero_grad(set_to_none=True)


def extract_lr_from_optimizer(
    optim: torch.optim.Optimizer, prefix: str
) -> Dict[str, float]:
    """
    Retrieves the learning rate values from an optimizer and returns them as a dictionary.
    """
    lr_stats = {}
    seen_pg_keys = {}
    for pg in optim.param_groups:
        lr = pg["lr"]
        name = _get_deduped_name(seen_pg_keys, pg.get("name", "pg"))
        key = f"{prefix}/{name}"
        assert key not in lr_stats
        lr_stats[key] = lr
    return lr_stats


def _get_deduped_name(seen_keys: Dict[str, int], name: str) -> str:
    if name not in seen_keys:
        seen_keys[name] = 0

    seen_keys[name] += 1
    return name + f":{seen_keys[name]-1}"
