#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import tempfile
import unittest

import torch
from torchtnt.utils.device import get_device_from_env
from torchtnt.utils.memory_snapshot_profiler import (
    MemorySnapshotParams,
    MemorySnapshotProfiler,
)
from torchtnt.utils.test_utils import skip_if_not_gpu


class MemorySnapshotProfilerGPUTest(unittest.TestCase):
    @skip_if_not_gpu
    def test_stop_step(self) -> None:
        """Test that a memory snapshot is saved when stop_step is reached."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_snapshot_profiler = MemorySnapshotProfiler(
                output_dir=temp_dir,
                memory_snapshot_params=MemorySnapshotParams(start_step=0, stop_step=2),
            )

            # initialize device & allocate memory for tensors
            device = get_device_from_env()
            a = torch.rand((1024, 1024), device=device)
            b = torch.rand((1024, 1024), device=device)
            _ = (a + b) * (a - b)

            memory_snapshot_profiler.step()

            # Check if the corresponding files exist
            save_dir = os.path.join(temp_dir, "step_2_rank0")

            pickle_dump_path = os.path.join(save_dir, "snapshot.pickle")
            trace_path = os.path.join(save_dir, "trace_plot.html")
            segment_plot_path = os.path.join(save_dir, "segment_plot.html")

            # after first step files do not exist
            self.assertFalse(os.path.exists(pickle_dump_path))
            self.assertFalse(os.path.exists(trace_path))
            self.assertFalse(os.path.exists(segment_plot_path))

            # after second step stop_step is reached and files should exist
            memory_snapshot_profiler.step()
            self.assertTrue(os.path.exists(pickle_dump_path))
            self.assertTrue(os.path.exists(trace_path))
            self.assertTrue(os.path.exists(segment_plot_path))
