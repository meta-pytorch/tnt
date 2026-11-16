#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import atexit
import logging
from typing import Any, Dict, List, Mapping, Optional, Union

from torch.utils.tensorboard import SummaryWriter
from torchtnt.utils.distributed import get_global_rank
from torchtnt.utils.loggers.anomaly_logger import AnomalyLogger, TrackedMetric
from torchtnt.utils.loggers.logger import Scalar

logger: logging.Logger = logging.getLogger(__name__)


class TensorBoardLogger(AnomalyLogger):
    """
    Simple logger for TensorBoard.

    On construction, the logger creates a new events file that logs
    will be written to.  If the environment variable `RANK` is defined,
    logger will only log if RANK = 0.

    Metrics may be tracked for anomaly detection if they are configured in the
    optional `tracked_metrics` argument. See :class:`torchtnt.utils.loggers.AnomalyLogger`
    for more details.

    Note:
        If using this logger with distributed training:

        - This logger should be constructed on all ranks
        - This logger can call collective operations
        - Logs will be written on rank 0 only
        - Logger must be constructed synchronously *after* initializing the distributed process group.

    Args:
        path (str): path to write logs to
        tracked_metrics: Optional list of TrackedMetric objects to track for anomaly detection.
        *args: Extra positional arguments to pass to SummaryWriter
        **kwargs: Extra keyword arguments to pass to SummaryWriter

    Examples::

        from torchtnt.utils.loggers import TensorBoardLogger
        logger = TensorBoardLogger(path="tmp/tb_logs")
        logger.log("accuracy", 23.56, 10)
        logger.close()
    """

    def __init__(
        self: TensorBoardLogger,
        path: str,
        tracked_metrics: Optional[List[TrackedMetric]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(tracked_metrics)
        self._writer: Optional[SummaryWriter] = None
        self._path: str = path
        self._rank: int = get_global_rank()

        if self._rank == 0:
            logger.info(
                f"TensorBoard SummaryWriter instantiated. Files will be stored in: {path}"
            )
            self._writer = SummaryWriter(log_dir=path, *args, **kwargs)
        else:
            logger.debug(
                f"Not logging metrics on this host because env RANK: {self._rank} != 0"
            )

        atexit.register(self.close)

    @property
    def writer(self: TensorBoardLogger) -> Optional[SummaryWriter]:
        return self._writer

    @property
    def path(self: TensorBoardLogger) -> str:
        return self._path

    def log_dict(
        self: TensorBoardLogger, payload: Mapping[str, Scalar], step: int
    ) -> None:
        """Add multiple scalar values to TensorBoard.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int): step value to record
        """

        if self._writer:
            for k, v in payload.items():
                self.log(k, v, step)

    def log(self: TensorBoardLogger, name: str, data: Scalar, step: int) -> None:
        """Add scalar data to TensorBoard.

        Args:
            name (string): tag name used to group scalars
            data (float/int/Tensor): scalar data to log
            step (int): step value to record
        """

        if self._writer:
            self._writer.add_scalar(name, data, global_step=step, new_style=True)

        super().log(name, data, step)

    def log_text(self: TensorBoardLogger, name: str, data: str, step: int) -> None:
        """Add text data to summary.

        Args:
            name (string): tag name used to identify data
            data (string): string to save
            step (int): step value to record
        """

        if self._writer:
            self._writer.add_text(name, data, global_step=step)

    def log_hparams(
        self: TensorBoardLogger, hparams: Dict[str, Scalar], metrics: Dict[str, Scalar]
    ) -> None:
        """Add hyperparameter data to TensorBoard.

        Args:
            hparams (dict): dictionary of hyperparameter names and corresponding values
            metrics (dict): dictionary of name of metric and corresponding values
        """

        if self._writer:
            self._writer.add_hparams(hparams, metrics)

    def log_image(self: TensorBoardLogger, *args: Any, **kwargs: Any) -> None:
        """Add image data to TensorBoard.


        Args:
            *args (Any): Positional arguments passed to SummaryWriter.add_image
            **kwargs(Any): Keyword arguments passed to SummaryWriter.add_image
        """
        writer = self._writer
        if writer:
            writer.add_image(*args, **kwargs)

    def log_images(self: TensorBoardLogger, *args: Any, **kwargs: Any) -> None:
        """Add batched image data to summary.

        Args:
            *args (Any): Positional arguments passed to SummaryWriter.add_images
            **kwargs(Any): Keyword arguments passed to SummaryWriter.add_images
        """
        writer = self._writer
        if writer:
            writer.add_images(*args, **kwargs)

    def log_figure(self: TensorBoardLogger, *args: Any, **kwargs: Any) -> None:
        """Add matplotlib figure to TensorBoard.

        Args:
            *args (Any): Positional arguments passed to SummaryWriter.add_figure
            **kwargs(Any): Keyword arguments passed to SummaryWriter.add_figure
        """
        writer = self._writer
        if writer:
            writer.add_figure(*args, **kwargs)

    def log_audio(self: TensorBoardLogger, *args: Any, **kwargs: Any) -> None:
        """Add audio data to TensorBoard.

        Args:
            *args (Any): Positional arguments passed to SummaryWriter.add_audio
            **kwargs (Any): Keyword arguments passed to SummaryWriter.add_audio
        """
        writer = self._writer
        if writer:
            writer.add_audio(*args, **kwargs)

    def log_scalars(
        self: TensorBoardLogger,
        main_tag: str,
        tag_scalar_dict: Dict[str, Union[float, int]],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> None:
        """Log multiple values to TensorBoard.
        Args:
            main_tag (string): Parent name for the tags
            tag_scalar_dict (dict): dictionary of tag name and scalar value
            global_step (int): global step value to record
            walltime (float): Optional override default walltime (time.time())
        Returns:
            None
        """
        if self._writer:
            self._writer.add_scalars(
                main_tag=main_tag,
                tag_scalar_dict=tag_scalar_dict,
                global_step=global_step,
                walltime=walltime,
            )

    def log_histogram(self: TensorBoardLogger, *args: Any, **kwargs: Any) -> None:
        """Compute and add histogram to TensorBoard.

        Args:
            *args (Any): Positional arguments passed to SummaryWriter.add_histogram
            **kwargs(Any): Keyword arguments passed to SummaryWriter.add_histogram
        """
        if self._writer:
            self._writer.add_histogram(*args, **kwargs)

    def log_histogram_raw(self: TensorBoardLogger, *args: Any, **kwargs: Any) -> None:
        """Add pre-computed histogram to TensorBoard.

        Args:
            *args (Any): Positional arguments passed to SummaryWriter.add_histogram_raw
            **kwargs(Any): Keyword arguments passed to SummaryWriter.add_histogram_raw
        """
        if self._writer:
            self._writer.add_histogram_raw(*args, **kwargs)

    def flush(self: TensorBoardLogger) -> None:
        """Writes pending logs to disk."""

        if self._writer:
            self._writer.flush()

    def close(self: TensorBoardLogger) -> None:
        """Close writer, flushing pending logs to disk.
        Logs cannot be written after `close` is called.
        """

        if self._writer:
            self._writer.close()
            self._writer = None
