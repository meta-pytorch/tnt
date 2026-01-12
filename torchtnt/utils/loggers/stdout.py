# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import atexit
import logging
import sys
from typing import Mapping, Optional

from torchtnt.utils.distributed import rank_zero_fn
from torchtnt.utils.loggers.logger import MetricLogger, Scalar
from torchtnt.utils.loggers.utils import scalar_to_float

logger: logging.Logger = logging.getLogger(__name__)


class StdoutLogger(MetricLogger):
    """
    Logger that prints metrics to stdout on rank 0. Each step is logged on a different line.
    Metrics belonging to the same step will be printed in the order they were logged on
    the same line. Step number is treated as an opaque identifier, successive steps do
    not have to be consecutive, but it is generally good practice to make them so.

    Args:
        precision (int): The number of digits to print after the decimal point. The default value is
        set to 4. The output will be rounded per the usual rounding rules.

    Example:
        from torchtnt.utils.loggers import StdoutLogger

        logger = StdoutLogger()
        logger.log(step=1, name="accuracy", data=0.982378)
        logger.log(step=1, name="loss", data=0.23112)
        logger.log_dict(step=2, payload={"accuracy": 0.99123, "loss": 0.18787})

        This will print the following to stdout in order:
        [Step 1] accuracy=0.9824, loss=0.2311
        [Step 2] accuracy=0.9912, loss=0.1879
    """

    def __init__(self, precision: int = 4) -> None:
        self._current_step: Optional[int] = None
        self._precision = precision
        logger.info("Logging metrics to stdout")
        atexit.register(self.close)

    def _start_new_step_if_needed(self, step: int) -> None:
        if self._current_step is None or step != self._current_step:
            self._current_step = step
            print(f"\n[Step {step}]", end="")

    def _log_metric(self, metric_name: str, metric_value: Scalar) -> None:
        metric_value = scalar_to_float(metric_value)
        print(f" {metric_name}={metric_value:.{self._precision}f}", end="", flush=True)

    @rank_zero_fn
    def log(self, name: str, data: Scalar, step: int) -> None:
        self._start_new_step_if_needed(step)
        self._log_metric(name, data)

    @rank_zero_fn
    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        self._start_new_step_if_needed(step)
        for k, v in payload.items():
            self._log_metric(k, v)

    def close(self) -> None:
        print("\n")
        sys.stdout.flush()
