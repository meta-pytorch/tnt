# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import importlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Type

import torch
from torch.distributed.fsdp import StateDictType as _StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch as _BackwardPrefetch,
    MixedPrecision as _MixedPrecision,
    ShardingStrategy as _ShardingStrategy,
)
from torchtnt.utils.precision import convert_precision_str_to_dtype


class ShardingStrategy:
    """Supported values for `ShardingStrategy <https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy>`_"""

    FULL_SHARD = "FULL_SHARD"
    SHARD_GRAD_OP = "SHARD_GRAD_OP"
    NO_SHARD = "NO_SHARD"
    HYBRID_SHARD = "HYBRID_SHARD"
    _HYBRID_SHARD_ZERO2 = "_HYBRID_SHARD_ZERO2"

    @staticmethod
    def to_native_sharding_strategy(value: str) -> _ShardingStrategy:
        """Convert a string to its PyTorch native ShardingStrategy."""
        if value not in [
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP,
            ShardingStrategy.NO_SHARD,
            ShardingStrategy.HYBRID_SHARD,
            ShardingStrategy._HYBRID_SHARD_ZERO2,
        ]:
            raise ValueError(f"Invalid ShardingStrategy '{value}'")

        return _ShardingStrategy[value]


class BackwardPrefetch:
    """Supported values for `BackwardPrefetch <https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.BackwardPrefetch>`_"""

    BACKWARD_PRE = "BACKWARD_PRE"
    BACKWARD_POST = "BACKWARD_POST"

    @staticmethod
    def to_native_backward_prefetch(value: str) -> _BackwardPrefetch:
        """Convert a string to its PyTorch native BackwardPrefetch."""
        if value not in [
            BackwardPrefetch.BACKWARD_PRE,
            BackwardPrefetch.BACKWARD_POST,
        ]:
            raise ValueError(f"Invalid BackwardPrefetch '{value}'")

        return _BackwardPrefetch[value]


class StateDictType:
    """Supported values for `StateDictType <https://pytorch.org/docs/stable/fsdp.html>`_"""

    FULL_STATE_DICT = "FULL_STATE_DICT"
    LOCAL_STATE_DICT = "LOCAL_STATE_DICT"
    SHARDED_STATE_DICT = "SHARDED_STATE_DICT"

    @staticmethod
    def to_native_state_dict_type(value: str) -> _StateDictType:
        """Convert a string to its PyTorch native StateDictType."""
        if value not in [
            StateDictType.FULL_STATE_DICT,
            StateDictType.LOCAL_STATE_DICT,
            StateDictType.SHARDED_STATE_DICT,
        ]:
            raise ValueError(f"Invalid StateDictType '{value}'")

        return _StateDictType[value]


def _to_dtype_or_none(x: Optional[str]) -> Optional[torch.dtype]:
    return convert_precision_str_to_dtype(x) if x else None


@dataclass
class MixedPrecision:
    """Supported values for `MixedPrecision <https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision>`_"""

    param_dtype: Optional[str] = None
    reduce_dtype: Optional[str] = None
    buffer_dtype: Optional[str] = None
    keep_low_precision_grads: bool = False
    cast_forward_inputs: bool = False
    cast_root_forward_inputs: bool = True
    _module_classes_to_ignore: Sequence[str] = (
        "torch.nn.modules.batchnorm._BatchNorm",
    )

    def to_native_mixed_precision(self) -> _MixedPrecision:
        """Convert this instance to its PyTorch native MixedPrecision."""

        # Convert string module classes to their corresponding types
        # e.g. "torch.nn.modules.batchnorm._BatchNorm" -> torch.nn.modules.batchnorm._BatchNorm
        target_types: List[Type[torch.nn.Module]] = []
        for type_str in self._module_classes_to_ignore:
            path, _, attr = type_str.rpartition(".")
            try:
                target_types.append(getattr(importlib.import_module(path), attr))
            except (AttributeError, ModuleNotFoundError) as e:
                raise ValueError(f"Invalid module class '{type_str}': {e}")
        module_classes_to_ignore: Sequence[Type[torch.nn.Module]] = target_types

        return _MixedPrecision(
            param_dtype=_to_dtype_or_none(self.param_dtype),
            reduce_dtype=_to_dtype_or_none(self.reduce_dtype),
            buffer_dtype=_to_dtype_or_none(self.buffer_dtype),
            keep_low_precision_grads=self.keep_low_precision_grads,
            cast_forward_inputs=self.cast_forward_inputs,
            cast_root_forward_inputs=self.cast_root_forward_inputs,
            _module_classes_to_ignore=module_classes_to_ignore,
        )
