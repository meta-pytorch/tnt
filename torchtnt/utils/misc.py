#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

import torch

_SEC_IN_DAY: int = 60 * 60 * 24


def days_to_secs(days: Optional[int]) -> Optional[int]:
    """Convert time from days to seconds"""
    if days is None:
        return None
    if days < 0:
        raise ValueError(f"days must be non-negative, but was given {days}")
    return days * _SEC_IN_DAY


def transfer_weights(src_module: torch.nn.Module, dst_module: torch.nn.Module) -> None:
    for src_param, dst_param in zip(src_module.parameters(), dst_module.parameters()):
        dst_param.detach().copy_(src_param.to(dst_param.device))


def transfer_batch_norm_stats(
    src_module: torch.nn.Module, dst_module: torch.nn.Module
) -> None:
    """
    Transfer batch norm statistics between two same models
    """
    src_batch_norm_modules = []
    dst_batch_norm_modules = []

    # fetch all batch norm modules for both
    for module in src_module.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            src_batch_norm_modules.append(module)

    for module in dst_module.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            dst_batch_norm_modules.append(module)

    if len(src_batch_norm_modules) != len(dst_batch_norm_modules):
        raise ValueError(
            "Modules must have same number of batch norm layers"
            f"Src module has {len(src_batch_norm_modules)}"
            f"Dst module has {len(dst_batch_norm_modules)}"
        )

    # copy batch norm statistics
    for src_batch_norm_module, dst_batch_norm_module in zip(
        src_batch_norm_modules, dst_batch_norm_modules
    ):
        # pyrefly: ignore [missing-attribute]
        dst_batch_norm_module.running_mean.detach().copy_(
            # pyrefly: ignore [missing-attribute]
            src_batch_norm_module.running_mean.to(
                # pyrefly: ignore [missing-attribute]
                dst_batch_norm_module.running_mean.device
            )
        )
        # pyrefly: ignore [missing-attribute]
        dst_batch_norm_module.running_var.detach().copy_(
            # pyrefly: ignore [missing-attribute]
            src_batch_norm_module.running_var.to(
                # pyrefly: ignore [missing-attribute]
                dst_batch_norm_module.running_var.device
            )
        )
        # pyrefly: ignore [missing-attribute]
        dst_batch_norm_module.num_batches_tracked.detach().copy_(
            # pyrefly: ignore [missing-attribute]
            src_batch_norm_module.num_batches_tracked.to(
                # pyrefly: ignore [missing-attribute]
                dst_batch_norm_module.num_batches_tracked.device
            )
        )
        dst_batch_norm_module.momentum = src_batch_norm_module.momentum
