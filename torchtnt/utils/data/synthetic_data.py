#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import torch
from torch.utils.data import Dataset
from torchtnt.utils.device import get_device_from_env


logger: logging.Logger = logging.getLogger(__name__)

TItem = TypeVar("TItem")


@dataclass
class AbstractRandomDataset(Dataset, abc.ABC, Generic[TItem]):
    """
    An abstract base class for random datasets.

    Intended for subclassing, this class provides the framework for implementing
    custom random datasets. Each subclass should provide a concrete implementation
    of the `_generate_random_item` method that produces a single random dataset
    item of type `TItem`.

    Attributes:
        size (int, default=100): The total number of items the dataset will contain.
    """

    size: int = field(default=100)

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"Size must be greater than zero. (Received {self.size})")
        logger.debug(f"Instantiated {self.__class__.__name__} with {self.size} items")

    def __len__(self) -> int:
        """
        Get the total number of items in the dataset.

        Returns:
            int: Total number of items.
        """
        return self.size

    # pyrefly: ignore [bad-override-param-name]
    def __getitem__(self, idx: int) -> TItem:
        """
        Fetch a dataset item by index.

        Args:
            idx (int): Index of the desired dataset item.

        Returns:
            TItem: A single random item of type `TItem`.

        Raises:
            IndexError: If the provided index is out of valid range.
        """
        if 0 <= idx < self.size:
            return self._generate_random_item()

        raise IndexError(f"Index {idx} out of range [0, {self.size-1}]")

    @abc.abstractmethod
    def _generate_random_item(self) -> TItem:
        """
        Abstract method to produce a random dataset item.

        Subclasses must override this to define their specific random item generation.

        Returns:
            TItem: A single random item of type `TItem`.
        """
        raise NotImplementedError(
            "Subclasses of AbstractRandomDataset should implement _generate_random_item."
        )


def generate_random_square_image_tensor(
    num_channels: int, side_length: int
) -> torch.Tensor:
    """
    Generate a random tensor with the given image dimensions.

    Args:
        num_channels (int): Number of channels for the random square image.
        side_length (int): Side length of the random square image.

    Returns:
        torch.Tensor: Randomly generated tensor with shape [num_channels, side_length, side_length].

    Raises:
        ValueError: If num_channels or side_length is not greater than zero.
    """
    if num_channels <= 0:
        raise ValueError(
            f"num_channels must be greater than zero. Received: {num_channels}"
        )
    if side_length <= 0:
        raise ValueError(
            f"side_length must be greater than zero. Received: {side_length}"
        )

    return torch.rand(
        num_channels, side_length, side_length, device=get_device_from_env()
    )
