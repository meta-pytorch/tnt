#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Mapping

from torchtnt.utils.loggers.logger import MetricLogger, Scalar
from torchtnt.utils.loggers.tensorboard import TensorBoardLogger

logger: logging.Logger = logging.getLogger(__name__)


class MultiTensorBoardLogger(TensorBoardLogger):
    """
    Simple logger wrapping a list of TensorBoard Loggers.

    Emits all log calls to each registered TensorBoard logger.

    Args:
        loggers: (List[TensorBoardLogger]) The TensorBoard Loggers to log to.

    Examples::

        from torchtnt.utils.loggers import MultiTensorBoardLogger TensorBoardLogger
        from <yourmodule>.logger import CustomTensorBoardLogger
        loggers = [TensorBoardLogger(path="tmp/tb_logs"), CustomTensorBoardLogger(uri="...")]
        logger = MultiTensorBoardLogger(loggers)
        logger.log("accuracy", 23.56, 10)
        logger.close()
    """

    def __init__(self, loggers: List[MetricLogger]) -> None:
        self._loggers = loggers

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        """Add multiple scalar values to TensorBoard.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int, Optional): step value to record
        """

        for logger in self._loggers:
            logger.log_dict(payload, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        """Add scalar data to TensorBoard.

        Args:
            name (string): tag name used to group scalars
            data (float/int/Tensor): scalar data to log
            step (int, optional): step value to record
        """
        for logger in self._loggers:
            logger.log(name, data, step)

    def log_text(self, name: str, data: str, step: int) -> None:
        """Add text data to summary.

        Args:
            name (string): tag name used to identify data
            data (string): string to save
            step (int): step value to record
        """
        for logger in self._loggers:
            if hasattr(logger, "log_text"):
                logger.log_text(name, data, step)

    def log_hparams(
        self, hparams: Dict[str, Scalar], metrics: Dict[str, Scalar]
    ) -> None:
        """Add hyperparameter data to TensorBoard.

        Args:
            hparams (dict): dictionary of hyperparameter names and corresponding values
            metrics (dict): dictionary of name of metric and corresponding values
        """

        for logger in self._loggers:
            if hasattr(logger, "log_hparams"):
                logger.log_hparams(hparams, metrics)

    def log_image(self, *args: Any, **kwargs: Any) -> None:
        """Add image data to TensorBoard.


        Args:
            *args (Any): Positional arguments passed to SummaryWriter.add_image
            **kwargs(Any): Keyword arguments passed to SummaryWriter.add_image
        """
        for logger in self._loggers:
            if hasattr(logger, "log_image"):
                logger.log_image(*args, **kwargs)

    def log_images(self, *args: Any, **kwargs: Any) -> None:
        """Add batched image data to summary.

        Args:
            *args (Any): Positional arguments passed to SummaryWriter.add_images
            **kwargs(Any): Keyword arguments passed to SummaryWriter.add_images
        """
        for logger in self._loggers:
            if hasattr(logger, "log_images"):
                logger.log_images(*args, **kwargs)

    def log_audio(self, *args: Any, **kwargs: Any) -> None:
        """Add audio data to TensorBoard.

        Args:
            *args (Any): Positional arguments passed to SummaryWriter.add_audio
            **kwargs (Any): Keyword arguments passed to SummaryWriter.add_audio
        """
        for logger in self._loggers:
            if hasattr(logger, "log_audio"):
                logger.log_audio(*args, **kwargs)

    def flush(self) -> None:
        """Writes pending logs to disk."""

        for logger in self._loggers:
            if hasattr(logger, "flush"):
                logger.flush()

    def close(self) -> None:
        """Close writer, flushing pending logs to disk.
        Logs cannot be written after `close` is called.
        """

        for logger in self._loggers:
            logger.close()
