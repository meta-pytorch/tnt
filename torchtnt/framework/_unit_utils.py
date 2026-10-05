# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import collections
import inspect
import logging
from typing import Callable, Dict, List, Tuple, TypeVar

import torch
import typing_extensions
from torchtnt.framework.state import State

_logger: logging.Logger = logging.getLogger(__name__)
T = TypeVar("T")


def _step_requires_iterator(step_func: Callable[[State, T], object]) -> bool:
    """
    Helper function to evaluate whether the get_next_X_batch method should pass the data iterator to the `X_step`
    functions, or whether get_next_X_batch should call `next(data_iter)` and pass a single batch to the step method.

    This is closely tied to the Unit's corresponding step function signature.
    """
    argspec = inspect.getfullargspec(step_func)
    annotations = argspec.annotations
    if "data" not in annotations:
        _logger.warning(
            f"Expected step function to have an annotated argument named ``data``. Found {annotations}."
        )
        return False
    annotated_type = annotations["data"]
    return typing_extensions.get_origin(annotated_type) is collections.abc.Iterator


def _find_optimizers_for_module(
    module: torch.nn.Module, optimizers: Dict[str, torch.optim.Optimizer]
) -> List[Tuple[str, torch.optim.Optimizer]]:
    """
    Given a module, returns a list of optimizers that are associated with it.
    """
    optimizer_list = []
    # Match by parameter identity across all param groups, not by data_ptr(). Under
    # FSDP1 with use_orig_params=True, a param outside a rank's local shard is a
    # shared size-0 tensor, so data_ptr() varies by rank and ranks can disagree on
    # the match, diverging on the collectives in FSDP.optim_state_dict.
    module_params = {id(param) for param in module.parameters()}
    for optim_name, optimizer in optimizers.items():
        optimizer_params = {
            id(param)
            for param_group in optimizer.param_groups
            for param in param_group["params"]
        }
        if module_params <= optimizer_params:
            optimizer_list.append((optim_name, optimizer))
    return optimizer_list
