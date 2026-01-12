#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import atexit
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from time import monotonic
from typing import Dict, Mapping

from torchtnt.utils.distributed import get_global_rank
from torchtnt.utils.loggers.logger import Scalar
from torchtnt.utils.loggers.utils import scalar_to_float


logger: logging.Logger = logging.getLogger(__name__)


class FileLogger(ABC):
    """
    Abstract file logger.

    Args:
            path (str): path to write logs to
            steps_before_flushing: (int): Number of steps to store in log before flushing
            log_all_ranks: (bool): Log all ranks if true, else log only on rank 0.
    """

    def __init__(
        self,
        path: str,
        steps_before_flushing: int,
        log_all_ranks: bool,
    ) -> None:
        self._path: str = path
        self._rank: int = get_global_rank()
        self._log_all_ranks = log_all_ranks
        self._log_buffer: OrderedDict[int, Dict[str, float]] = OrderedDict()
        self._len_before_flush: int = 0
        self._steps_before_flushing: int = steps_before_flushing

        if self._rank == 0 or log_all_ranks:
            logger.info(f"Logging metrics to path: {path}")
        else:
            logger.debug(
                f"Not logging metrics on this host because host rank is {self._rank} != 0"
            )
        atexit.register(self.close)

    @property
    def path(self) -> str:
        return self._path

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        """Add multiple scalar values.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int): step value to record
        """

        for k, v in payload.items():
            self.log(k, v, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        """Log scalar data to file.

        Args:
            name (string): a unique name to group scalars
            data (float/int/Tensor): scalar data to log
            step (int): step value to record
        """

        if self._rank == 0 or self._log_all_ranks:
            self._log_buffer.setdefault(step, {})[name] = scalar_to_float(data)
            self._log_buffer[step]["step"] = step
            self._log_buffer[step]["time"] = monotonic()

        if (
            len(self._log_buffer) - self._len_before_flush
            >= self._steps_before_flushing
        ):
            self.flush()
            self._len_before_flush = len(self._log_buffer)

    @abstractmethod
    def flush(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...
