#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import tempfile
import unittest

import torch
from torchtnt.utils.device import get_device_from_env
from torchtnt.utils.oom import log_memory_snapshot
from torchtnt.utils.test_utils import skip_if_not_gpu


class OomGPUTest(unittest.TestCase):
    @skip_if_not_gpu
    def test_log_memory_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Record history
            torch.cuda.memory._record_memory_history(enabled="all", max_entries=10000)

            # initialize device & allocate memory for tensors
            device = get_device_from_env()
            a = torch.rand((1024, 1024), device=device)
            b = torch.rand((1024, 1024), device=device)
            _ = (a + b) * (a - b)

            # save a snapshot
            log_memory_snapshot(temp_dir, "foo")

            # Check if the corresponding files exist
            save_dir = os.path.join(temp_dir, "foo_rank0")

            pickle_dump_path = os.path.join(save_dir, "snapshot.pickle")
            self.assertTrue(os.path.exists(pickle_dump_path))

            trace_path = os.path.join(save_dir, "trace_plot.html")
            self.assertTrue(os.path.exists(trace_path))

            segment_plot_path = os.path.join(save_dir, "segment_plot.html")
            self.assertTrue(os.path.exists(segment_plot_path))
