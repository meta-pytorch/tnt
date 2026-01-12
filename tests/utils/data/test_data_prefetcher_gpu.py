#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest
from typing import Tuple

import torch
from torch.utils.data import Dataset, TensorDataset
from torchtnt.utils.data.data_prefetcher import CudaDataPrefetcher
from torchtnt.utils.test_utils import skip_if_not_gpu

Batch = Tuple[torch.Tensor, torch.Tensor]


class DataPrefetcherGPUTest(unittest.TestCase):
    def _generate_dataset(self, num_samples: int, input_dim: int) -> Dataset[Batch]:
        """Returns a dataset of random inputs and labels for binary classification."""
        data = torch.randn(num_samples, input_dim)
        labels = torch.randint(low=0, high=2, size=(num_samples,))
        return TensorDataset(data, labels)

    @skip_if_not_gpu
    def test_cuda_data_prefetcher(self) -> None:
        device = torch.device("cuda:0")

        num_samples = 12
        batch_size = 4
        dataloader = torch.utils.data.DataLoader(
            self._generate_dataset(num_samples, 2), batch_size=batch_size
        )

        num_prefetch_batches = 2
        data_prefetcher = CudaDataPrefetcher(dataloader, device, num_prefetch_batches)
        self.assertEqual(num_prefetch_batches, data_prefetcher.num_prefetch_batches)

        # make sure data_prefetcher has same number of samples as original dataloader
        num_batches_in_data_prefetcher = 0
        for inputs, targets in data_prefetcher:
            num_batches_in_data_prefetcher += 1
            # len(inputs) should equal the batch size
            self.assertEqual(len(inputs), batch_size)
            self.assertEqual(len(targets), batch_size)
            # make sure batch is on correct device
            self.assertEqual(inputs.device, device)
            self.assertEqual(targets.device, device)

        self.assertEqual(num_batches_in_data_prefetcher, num_samples / batch_size)
