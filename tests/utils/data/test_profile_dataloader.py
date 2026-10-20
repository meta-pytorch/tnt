#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Iterator

import torch
from torch.profiler import ProfilerActivity
from torchtnt.utils.data.profile_dataloader import profile_dataloader
from torchtnt.utils.env import init_from_env


class DummyIterable:
    def __init__(self, count: int) -> None:
        self.count: int = count

    def __iter__(self) -> Iterator[int]:
        for i in range(self.count):
            yield i


class ProfileDataLoaderTest(unittest.TestCase):
    def test_profile_dataloader(self) -> None:
        max_length = 10
        iterable = DummyIterable(max_length)
        with _get_torch_profiler() as p:
            timer = profile_dataloader(iterable, p)
        self.assertEqual(len(timer.recorded_durations["next(iter)"]), max_length)

    def test_profile_dataloader_max_steps(self) -> None:
        max_length = 10
        max_steps = 5
        iterable = DummyIterable(max_length)
        with _get_torch_profiler() as p:
            timer = profile_dataloader(iterable, p, max_steps=max_steps)
        self.assertEqual(len(timer.recorded_durations["next(iter)"]), max_steps)

    def test_profile_dataloader_profiler(self) -> None:
        max_length = 10
        iterable = DummyIterable(max_length)
        with _get_torch_profiler() as p:
            timer = profile_dataloader(iterable, p)
        self.assertEqual(len(timer.recorded_durations["next(iter)"]), max_length)

    def test_profile_dataloader_device(self) -> None:
        device = init_from_env()
        max_length = 10
        iterable = DummyIterable(max_length)
        with _get_torch_profiler() as p:
            timer = profile_dataloader(iterable, p, device=device)
        self.assertEqual(len(timer.recorded_durations["next(iter)"]), max_length)
        self.assertEqual(
            len(timer.recorded_durations["copy_data_to_device"]), max_length
        )


def _get_torch_profiler() -> torch.profiler.profile:
    profiler_schedule = torch.profiler.schedule(
        wait=0,
        warmup=1,
        active=1,
    )
    return torch.profiler.profile(
        activities=[ProfilerActivity.CPU],
        schedule=profiler_schedule,
    )
