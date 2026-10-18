# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Iterable, Iterator, List, TypeVar

import torch
from torchtnt.utils.device import copy_data_to_device, record_data_in_stream

Batch = TypeVar("Batch")


class CudaDataPrefetcher(Iterator[Batch]):
    r"""CudaDataPrefetcher prefetches batches and moves them to the device.

    This class can be used to interleave data loading, host-to-device copies, and computation more effectively.

    Args:

        data_iterable: an Iterable containing the data to use for CudaDataPrefetcher construction
        device: the device to which data should be moved
        num_prefetch_batches: number of batches to prefetch

    Note:

        We recommend users leverage memory pinning when constructing their dataloader:
        https://pytorch.org/docs/stable/data.html#memory-pinning.

    Example::

        dataloader = ...
        device = torch.device("cuda")
        num_prefetch_batches = 2
        data_prefetcher = CudaDataPrefetcher(dataloader, device, num_prefetch_batches)
        for batch in data_prefetcher:
            # batch is already on device
            # operate on batch
    """

    def __init__(
        self,
        data_iterable: Iterable[Batch],
        device: torch.device,
        num_prefetch_batches: int = 1,
    ) -> None:
        if device.type != "cuda":
            raise ValueError(
                "`CudaDataPrefetcher` expects a CUDA device, but got "
                f"device type {device.type}."
            )
        if num_prefetch_batches < 1:
            raise ValueError(
                "`num_prefetch_batches` must be greater than 0. Got "
                f"{num_prefetch_batches}."
            )
        self.data_iterable = data_iterable
        self.device = device
        self.num_prefetch_batches = num_prefetch_batches
        self._reset()

    def _reset(self) -> None:
        self._prefetched: bool = False
        self._batches: List[Batch] = []
        self._events: List[torch.cuda.Event] = []
        self._prefetch_stream = torch.cuda.Stream()
        self.data_iter = iter(self.data_iterable)

    def _prefetch(self) -> None:
        for _ in range(self.num_prefetch_batches):
            try:
                self._fetch_next_batch(self.data_iter)
            except StopIteration:
                return

    def _fetch_next_batch(self, data_iter: Iterator[Batch]) -> None:
        try:
            next_batch = next(data_iter)
        except StopIteration:
            return

        event = torch.cuda.Event()
        with torch.cuda.stream(self._prefetch_stream):
            next_batch = copy_data_to_device(next_batch, self.device, non_blocking=True)

        self._batches.append(next_batch)
        event.record(self._prefetch_stream)
        self._events.append(event)

    def __iter__(self) -> "CudaDataPrefetcher[Batch]":
        self._reset()
        return self

    def __next__(self) -> Batch:
        if not self._prefetched:
            self._prefetch()
            self._prefetched = True
        # wait for the prefetch stream to complete host to device copy
        if self._events:
            event = self._events.pop(0)
            event.wait()
        else:
            raise StopIteration

        if self._batches:
            # there are pre-fetched batches already from a previous `prefetching` call.
            # consume one
            batch = self._batches.pop(0)
            # the batch was allocated on the prefetch stream; tell the caching
            # allocator it is used on the consumer stream so its memory is not
            # reused while consumer kernels may still read it
            record_data_in_stream(batch, torch.cuda.current_stream())
            # refill the consumed batch
            try:
                self._fetch_next_batch(self.data_iter)
            except StopIteration:
                pass
        else:
            # the iterator is empty
            raise StopIteration

        return batch
