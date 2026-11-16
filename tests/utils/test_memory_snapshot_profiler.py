#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import tempfile
import unittest

from torchtnt.utils.memory_snapshot_profiler import (
    MemorySnapshotParams,
    MemorySnapshotProfiler,
)


class MemorySnapshotProfilerTest(unittest.TestCase):
    def test_validation(self) -> None:
        """Test parameter validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "start_step must be nonnegative."):
                _ = MemorySnapshotProfiler(
                    output_dir=temp_dir,
                    memory_snapshot_params=MemorySnapshotParams(
                        start_step=-1, stop_step=0
                    ),
                )
            with self.assertRaisesRegex(
                ValueError, "stop_step must be specified when start_step is set."
            ):
                _ = MemorySnapshotProfiler(
                    output_dir=temp_dir,
                    memory_snapshot_params=MemorySnapshotParams(
                        start_step=2, stop_step=None
                    ),
                )
            with self.assertRaisesRegex(ValueError, "start_step must be < stop_step."):
                _ = MemorySnapshotProfiler(
                    output_dir=temp_dir,
                    memory_snapshot_params=MemorySnapshotParams(
                        start_step=2, stop_step=0
                    ),
                )
            with self.assertRaisesRegex(ValueError, "stop_step must be positive."):
                _ = MemorySnapshotProfiler(
                    output_dir=temp_dir,
                    memory_snapshot_params=MemorySnapshotParams(stop_step=0),
                )
            with self.assertRaisesRegex(
                ValueError,
                "stop_step must be enabled with either start_step or enable_oom_observer.",
            ):
                _ = MemorySnapshotProfiler(
                    output_dir=temp_dir,
                    memory_snapshot_params=MemorySnapshotParams(
                        stop_step=2, enable_oom_observer=False
                    ),
                )
            with self.assertRaisesRegex(
                ValueError,
                "At least one of start_step/stop_step or enable_oom_observer must be set.",
            ):
                _ = MemorySnapshotProfiler(
                    output_dir=temp_dir,
                    memory_snapshot_params=MemorySnapshotParams(
                        start_step=None, stop_step=None, enable_oom_observer=False
                    ),
                )
