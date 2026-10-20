#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
import random
from datetime import timedelta
from typing import Optional, Union

import numpy as np
import torch
from torch.distributed.constants import default_pg_timeout
from torchtnt.utils.device import get_device_from_env, set_float32_precision
from torchtnt.utils.distributed import (
    get_file_init_method,
    get_process_group_backend_from_device,
    get_tcp_init_method,
)
from typing_extensions import Literal

_log: logging.Logger = logging.getLogger(__name__)


def _check_dist_env() -> bool:
    """
    Check if all environment variables required to initialize torch.distributed are set
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    """
    env_required = (
        os.environ.get("MASTER_PORT"),
        os.environ.get("MASTER_ADDR"),
        os.environ.get("WORLD_SIZE"),
        os.environ.get("RANK"),
    )
    return all(env is not None for env in env_required)


def init_from_env(
    *,
    device_type: Optional[str] = None,
    dist_init_method_type: Literal["env", "tcp", "file"] = "env",
    pg_backend: Optional[str] = None,
    pg_timeout: timedelta = default_pg_timeout,
    float32_matmul_precision: str = "high",
) -> torch.device:
    """Utility function that initializes the device and process group, if applicable.

    The global process group is initialized only if:
        - torch.distributed is available and has not already been initialized
        - the program has been launched on multiple processes

    This is intended as a convenience to include at the beginning of scripts that follow
    a SPMD-style execution model.


    Args:
        device_type (str, optional): Device type to initialize. If None, device will be initialized
                                  based on environment
        dist_init_method_type (str, optional): Method to initialize the process group. Must be one of "env", "tcp", or "file".
            For more information, see here: https://pytorch.org/docs/stable/distributed.html#initialization
        pg_backend (str, optional): The process group backend to use. If None, it will use the
                                    default process group backend from the device
        pg_timeout (timedelta, optional): Timeout for operations executed against the process
                                          group. Default value equals 30 minutes
        float32_matmul_precision (str, optional): The setting for torch's precision of matrix multiplications.

    Returns:
        The current device.
    """
    device = torch.device("cpu") if device_type == "cpu" else get_device_from_env()

    if device_type is not None and device.type != device_type:
        raise RuntimeError(
            f"Device type is specified to {device_type} but got {device.type} from env"
        )

    if _check_dist_env():
        if not torch.distributed.is_available():
            _log.warning(
                "torch.distributed is not available. Skipping initializing the process group."
            )
            return device
        if torch.distributed.is_initialized():
            _log.warning(
                "torch.distributed is already initialized. Skipping initializing the process group."
            )
            return device
        pg_backend = (
            pg_backend
            if pg_backend is not None
            else get_process_group_backend_from_device(device)
        )
        init_method: Optional[str] = None
        if dist_init_method_type == "tcp":
            init_method = get_tcp_init_method()
        elif dist_init_method_type == "file":
            init_method = get_file_init_method()
        torch.distributed.init_process_group(
            init_method=init_method, backend=pg_backend, timeout=pg_timeout
        )
    set_float32_precision(float32_matmul_precision)
    return device


def seed(seed: int, deterministic: Optional[Union[str, int]] = None) -> None:
    """Function that sets seed for pseudo-random number generators across commonly used libraries.

    This seeds PyTorch, NumPy, and the python.random module.
    For more details, see https://pytorch.org/docs/stable/notes/randomness.html.

    Args:
        seed: the integer value seed.
        deterministic: Controls determinism settings within PyTorch.
            If `None`, don't set any PyTorch global values.
            If "default" or 0, don't error or warn on nondeterministic operations and additionally enable PyTorch CuDNN benchmark.
            If "warn" or 1, warn on nondeterministic operations and disable PyTorch CuDNN benchmark.
            If "error" or 2, error on nondeterministic operations and disable PyTorch CuDNN benchmark.
            For more details, see :func:`torch.set_deterministic_debug_mode` and
            https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms.

    Raises:
        ValueError
            If the input seed value is outside the required range.
    """
    max_val = np.iinfo(np.uint32).max
    min_val = np.iinfo(np.uint32).min
    if seed < min_val or seed > max_val:
        raise ValueError(
            f"Invalid seed value provided: {seed}. Value must be in the range [{min_val}, {max_val}]"
        )
    _log.info(f"Setting seed to {seed}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic is not None:
        _log.info(f"Setting deterministic debug mode to {deterministic}")
        torch.set_deterministic_debug_mode(deterministic)
        deterministic_debug_mode = torch.get_deterministic_debug_mode()
        if deterministic_debug_mode == 0:
            _log.info("Disabling cuDNN deterministic mode")
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            _log.info("Enabling cuDNN deterministic mode")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # reference: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if deterministic_debug_mode == 2:
            set_float32_precision("highest")
