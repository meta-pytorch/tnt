#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import ctypes
import sys
import unittest
import uuid
from contextlib import contextmanager
from functools import wraps
from io import StringIO
from typing import Callable, Generator, Optional, TextIO, Tuple, TypeVar

import torch
import torch.distributed.launcher as pet
from pyre_extensions import ParameterSpecification


TParams = ParameterSpecification("TParams")
TReturn = TypeVar("TReturn")


def get_pet_launch_config(nproc: int) -> pet.LaunchConfig:
    """
    Initialize pet.LaunchConfig for single-node, multi-rank functions.

    Args:
        nproc: The number of processes to launch.

    Returns:
        An instance of pet.LaunchConfig for single-node, multi-rank functions.

    Example:
        >>> from torch.distributed import launcher
        >>> launch_config = get_pet_launch_config(nproc=8)
        >>> launcher.elastic_launch(config=launch_config, entrypoint=train)()
    """
    return pet.LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=nproc,
        run_id=str(uuid.uuid4()),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        max_restarts=0,
        monitor_interval=1,
    )


def is_asan() -> bool:
    """Determines if the Python interpreter is running with ASAN"""
    return hasattr(ctypes.CDLL(""), "__asan_init")


def is_tsan() -> bool:
    """Determines if the Python interpreter is running with TSAN"""
    return hasattr(ctypes.CDLL(""), "__tsan_init")


def is_asan_or_tsan() -> bool:
    return is_asan() or is_tsan()


def skip_if_asan(
    func: Callable[TParams, TReturn],
) -> Callable[TParams, Optional[TReturn]]:
    """Skip test run if we are in ASAN mode."""

    @wraps(func)
    def wrapper(*args: TParams.args, **kwargs: TParams.kwargs) -> Optional[TReturn]:
        if is_asan_or_tsan():
            print("Skipping test run since we are in ASAN mode.")
            return
        return func(*args, **kwargs)

    return wrapper


@contextmanager
def captured_output() -> Generator[Tuple[TextIO, TextIO], None, None]:
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


"""Decorator for tests to ensure running on a GPU."""
skip_if_not_gpu: Callable[..., Callable[..., object]] = unittest.skipUnless(
    torch.cuda.is_available(), "Skipping test since GPU is not available"
)

"""Decorator for tests to ensure running when distributed is available."""
skip_if_not_distributed: Callable[..., Callable[..., object]] = unittest.skipUnless(
    torch.distributed.is_available(), "Skipping test since distributed is not available"
)
