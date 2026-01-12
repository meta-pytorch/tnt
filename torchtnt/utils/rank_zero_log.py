#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Any, Optional

from packaging.version import Version
from torchtnt.utils.distributed import get_global_rank
from torchtnt.utils.version import get_python_version

_LOGGER: logging.Logger = logging.getLogger(__name__)


def rank_zero_print(*args: Any, **kwargs: Any) -> None:
    """Call print function only from rank 0."""
    if get_global_rank() != 0:
        return
    print(*args, **kwargs)


def rank_zero_debug(
    *args: Any, logger: Optional[logging.Logger] = None, **kwargs: Any
) -> None:
    """Log debug message only from rank 0."""
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    if _supports_stacklevel():
        kwargs["stacklevel"] = 2
    logger.debug(*args, **kwargs)


def rank_zero_info(
    *args: Any, logger: Optional[logging.Logger] = None, **kwargs: Any
) -> None:
    """Log info message only from rank 0."""
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    if _supports_stacklevel():
        kwargs["stacklevel"] = 2
    logger.info(*args, **kwargs)


def rank_zero_warn(
    *args: Any, logger: Optional[logging.Logger] = None, **kwargs: Any
) -> None:
    """Log warn message only from rank 0."""
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    if _supports_stacklevel():
        kwargs["stacklevel"] = 2
    logger.warning(*args, **kwargs)


def rank_zero_error(
    *args: Any, logger: Optional[logging.Logger] = None, **kwargs: Any
) -> None:
    """Log error message only from rank 0."""
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    if _supports_stacklevel():
        kwargs["stacklevel"] = 2
    logger.error(*args, **kwargs)


def rank_zero_critical(
    *args: Any, logger: Optional[logging.Logger] = None, **kwargs: Any
) -> None:
    """Log critical message only from rank 0."""
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    if _supports_stacklevel():
        kwargs["stacklevel"] = 2
    logger.critical(*args, **kwargs)


def _supports_stacklevel() -> bool:
    return get_python_version() >= Version("3.8.0")
