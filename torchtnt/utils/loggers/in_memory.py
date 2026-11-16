#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import atexit
import logging
from collections import OrderedDict
from time import monotonic
from typing import Dict, Mapping

from torchtnt.utils.loggers.logger import MetricLogger, Scalar
from torchtnt.utils.loggers.utils import scalar_to_float

logger: logging.Logger = logging.getLogger(__name__)


class InMemoryLogger(MetricLogger):
    """
    Simple logger that buffers data in-memory.

    Example:
            from torchtnt.utils.loggers import InMemoryLogger
            logger = InMemoryLogger()
            logger.log("accuracy", 23.56, 10)
            logger.close()
    """

    def __init__(self) -> None:
        self._log_buffer: OrderedDict[int, Dict[str, float]] = OrderedDict()
        logger.info("Logging metrics in-memory")
        atexit.register(self.close)

    @property
    def log_buffer(self) -> Dict[int, Dict[str, float]]:
        """Directly access the buffer."""

        return self._log_buffer

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        """Add multiple scalar values.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int): step value to record
        """

        for k, v in payload.items():
            self.log(k, v, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        """Log scalar data to the in-memory buffer.

        Args:
            name (string): a unique name to group scalars
            data (float/int/Tensor): scalar data to log
            step (int): step value to record
        """

        self._log_buffer.setdefault(step, {})[name] = scalar_to_float(data)
        self._log_buffer[step]["step"] = step
        self._log_buffer[step]["time"] = monotonic()

    def flush(self) -> None:
        print(self._log_buffer)

    def close(self) -> None:
        self._log_buffer.clear()
