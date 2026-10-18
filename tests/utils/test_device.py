#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from typing import Any, cast, List
from unittest import mock

import torch
from torchtnt.utils.device import (
    copy_data_to_device,
    get_device_from_env,
    get_nvidia_smi_gpu_stats,
    get_psutil_cpu_stats,
)


class _RecordingData:
    """Tensor-like leaf that records the streams passed to ``record_stream``."""

    def __init__(self) -> None:
        self.recorded_streams: List[Any] = []

    def to(self, *args: Any, **kwargs: Any) -> "_RecordingData":
        return self

    def record_stream(self, stream: Any) -> None:
        self.recorded_streams.append(stream)


class DeviceTest(unittest.TestCase):
    def test_copy_data_to_device_records_stream_on_nested_data(self) -> None:
        Pair = namedtuple("Pair", ["a", "b"])

        @dataclass
        class Batch:
            x: Any
            y: Any = field(init=False)

        leaves = [_RecordingData() for _ in range(9)]
        batch = Batch(x=leaves[7])
        batch.y = leaves[8]
        data = {
            "list": [leaves[0], leaves[1]],
            "tuple": (leaves[2],),
            "namedtuple": Pair(a=leaves[3], b=leaves[4]),
            "defaultdict": defaultdict(list, {"k": leaves[5]}),
            "nested": {"leaf": leaves[6]},
            "dataclass": batch,
        }
        stream = cast(torch.cuda.Stream, object())

        copy_data_to_device(data, torch.device("cpu"), stream_to_record=stream)

        for leaf in leaves:
            self.assertEqual(leaf.recorded_streams, [stream])

    def test_copy_data_to_device_no_stream_does_not_record(self) -> None:
        leaf = _RecordingData()
        copy_data_to_device({"a": [leaf]}, torch.device("cpu"))
        self.assertEqual(leaf.recorded_streams, [])

    def test_get_cpu_device(self) -> None:
        device = get_device_from_env()
        self.assertEqual(device.type, "cpu")
        self.assertEqual(device.index, None)

    def test_get_cpu_stats(self) -> None:
        """Get CPU stats, check that values are populated."""
        cpu_stats = get_psutil_cpu_stats()
        # Check that percentages are between 0 and 100
        self.assertGreaterEqual(cpu_stats["cpu_vm_percent"], 0)
        self.assertLessEqual(cpu_stats["cpu_vm_percent"], 100)
        self.assertGreaterEqual(cpu_stats["cpu_percent"], 0)
        self.assertLessEqual(cpu_stats["cpu_percent"], 100)
        self.assertGreaterEqual(cpu_stats["cpu_swap_percent"], 0)
        self.assertLessEqual(cpu_stats["cpu_swap_percent"], 100)

    def test_get_gpu_stats(self) -> None:
        """Get Nvidia GPU stats, check that values are populated."""
        device = torch.device("cuda:0")

        with mock.patch("shutil.which"), mock.patch(
            "torchtnt.utils.device.subprocess.run"
        ) as subprocess_run_mock:
            subprocess_run_mock.return_value.stdout = "0, 0, 0, 2, 16273, 38, 15"
            gpu_stats = get_nvidia_smi_gpu_stats(device)

        # Check that percentages are between 0 and 100
        self.assertGreaterEqual(gpu_stats["utilization_gpu_percent"], 0)
        self.assertLessEqual(gpu_stats["utilization_gpu_percent"], 100)
        self.assertGreaterEqual(gpu_stats["utilization_memory_percent"], 0)
        self.assertLessEqual(gpu_stats["utilization_memory_percent"], 100)
        self.assertGreaterEqual(gpu_stats["fan_speed_percent"], 0)
        self.assertLessEqual(gpu_stats["fan_speed_percent"], 100)

        # Check that values are greater than zero
        self.assertGreaterEqual(gpu_stats["memory_used_mb"], 0)
        self.assertGreaterEqual(gpu_stats["memory_free_mb"], 0)
        self.assertGreaterEqual(gpu_stats["temperature_gpu_celsius"], 0)
        self.assertGreaterEqual(gpu_stats["temperature_memory_celsius"], 0)
