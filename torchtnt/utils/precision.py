#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Mapping, Optional

import torch
from torch.amp.grad_scaler import GradScaler

_DTYPE_STRING_TO_DTYPE_MAPPING: Mapping[str, Optional[torch.dtype]] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": None,
}


def convert_precision_str_to_dtype(precision: str) -> Optional[torch.dtype]:
    """
    Converts precision as a string to a torch.dtype

    Args:
        precision: string containing the precision

    Raises:
        ValueError if an invalid precision string is passed.

    """
    if precision not in _DTYPE_STRING_TO_DTYPE_MAPPING:
        raise ValueError(
            f"Precision {precision} not supported. Please use one of {list(_DTYPE_STRING_TO_DTYPE_MAPPING.keys())}"
        )
    return _DTYPE_STRING_TO_DTYPE_MAPPING[precision]


def get_grad_scaler_from_precision(
    precision: torch.dtype,
    *,
    is_fsdp1_module: Optional[bool] = False,
    grad_scaler_kwargs: Optional[dict[str, Any]] = None,
) -> Optional[GradScaler]:
    """
    Returns the correct grad scaler to use based on the precision and whether
    or not the model is FSDP. FSDP required it's own sharded grad scaler. FSDP2 uses
    the original grad scaler (amp.grad_scaler). See https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md

    Args:
        precision: the precision being used
        is_fsdp1_module: whether the grad scaler is for an FSDP1 module
        grad_scaler_kwargs: optional parameters for configuring the grad scaler
            (init_scale, growth_factor, backoff_factor, growth_interval)

    Returns:
        The appropriate grad scaler to use, ``None`` if no grad scaler should be used.
    """

    if precision == torch.float16:

        if grad_scaler_kwargs is None:
            grad_scaler_kwargs = {}

        if is_fsdp1_module:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            return ShardedGradScaler(
                **grad_scaler_kwargs,
            )
        else:
            return GradScaler(
                device="cuda",
                **grad_scaler_kwargs,
            )
    return None
