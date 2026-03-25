#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import csv
import logging
from threading import Thread
from typing import Dict, List, Optional

from fsspec import open as fs_open
from torchtnt.utils.loggers.file import FileLogger
from torchtnt.utils.loggers.logger import MetricLogger

logger: logging.Logger = logging.getLogger(__name__)


class CSVLogger(FileLogger, MetricLogger):
    """
    CSV file logger. CSV headers are time, step, and names passed to `log`.

    Args:
        path (str): path to write logs to
        steps_before_flushing: (int, optional): Number of steps to buffer in logger before flushing
        log_all_ranks: (bool, optional): Log all ranks if true, else log only on rank 0.
        async_write: (bool, optional): Whether to write asynchronously or not. Defaults to False.
    """

    def __init__(
        self,
        path: str,
        steps_before_flushing: int = 100,
        log_all_ranks: bool = False,
        async_write: bool = False,
    ) -> None:
        super().__init__(path, steps_before_flushing, log_all_ranks)

        self._async_write = async_write
        self._thread: Optional[Thread] = None

    def flush(self) -> None:
        if self._rank == 0 or self._log_all_ranks:
            buffer = self._log_buffer
            if not buffer:
                logger.debug("No logs to write.")
                return

            if self._thread:
                # ensure previous thread is completed before next write
                self._thread.join()

            data_list = list(buffer.values())
            if not self._async_write:
                _write_csv(self.path, data_list)
                return

            self._thread = Thread(
                target=_write_csv,
                args=(self.path, data_list),
                name="CSVAsyncWriter",
            )
            self._thread.start()

    def close(self) -> None:
        # toggle off async writing for final flush
        self._async_write = False
        self.flush()


def _write_csv(path: str, data_list: List[Dict[str, float]]) -> None:
    with fs_open(path, "w") as f:
        w = csv.DictWriter(f, data_list[0].keys())
        w.writeheader()
        w.writerows(data_list)
