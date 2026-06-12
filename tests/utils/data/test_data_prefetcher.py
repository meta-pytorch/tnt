#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Tuple
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data.dataset import Dataset, TensorDataset
from torchtnt.utils.data.data_prefetcher import CudaDataPrefetcher

Batch = Tuple[torch.Tensor, torch.Tensor]


class DataPrefetcherTest(unittest.TestCase):
    def _generate_dataset(self, num_samples: int, input_dim: int) -> Dataset[Batch]:
        """Returns a dataset of random inputs and labels for binary classification."""
        data = torch.randn(num_samples, input_dim)
        labels = torch.randint(low=0, high=2, size=(num_samples,))
        # pyrefly: ignore [bad-return]
        return TensorDataset(data, labels)

    def test_device_data_prefetcher(self) -> None:
        device = torch.device("cpu")

        num_samples = 12
        batch_size = 4
        dataloader = torch.utils.data.DataLoader(
            self._generate_dataset(num_samples, 2), batch_size=batch_size
        )

        num_prefetch_batches = 2
        with self.assertRaisesRegex(ValueError, "expects a CUDA device"):
            _ = CudaDataPrefetcher(dataloader, device, num_prefetch_batches)

    @patch("torch.cuda.Stream")
    def test_num_prefetch_batches_data_prefetcher(self, mock_stream: MagicMock) -> None:
        device = torch.device("cuda:0")

        num_samples = 12
        batch_size = 4
        dataloader = torch.utils.data.DataLoader(
            self._generate_dataset(num_samples, 2), batch_size=batch_size
        )

        with self.assertRaisesRegex(
            ValueError, "`num_prefetch_batches` must be greater than 0"
        ):
            _ = CudaDataPrefetcher(dataloader, device, num_prefetch_batches=-1)

        with self.assertRaisesRegex(
            ValueError, "`num_prefetch_batches` must be greater than 0"
        ):
            _ = CudaDataPrefetcher(dataloader, device, num_prefetch_batches=0)

        # no exceptions raised
        _ = CudaDataPrefetcher(dataloader, device, num_prefetch_batches=1)
        _ = CudaDataPrefetcher(dataloader, device, num_prefetch_batches=2)

        # Check that CUDA streams were created
        self.assertEqual(mock_stream.call_count, 2)
