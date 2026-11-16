#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from torchtnt.utils.oom import (
    _bytes_to_mb_gb,
    is_out_of_cpu_memory,
    is_out_of_cuda_memory,
    is_out_of_memory_error,
)


class OomTest(unittest.TestCase):
    def test_is_out_of_cpu_memory(self) -> None:
        """Test CPU OOM error detection."""
        cpu_oom_error = RuntimeError("DefaultCPUAllocator: can't allocate memory")
        self.assertTrue(is_out_of_cpu_memory(cpu_oom_error))
        not_cpu_oom_error = RuntimeError("RuntimeError: blah")
        self.assertFalse(is_out_of_cpu_memory(not_cpu_oom_error))

    def test_is_out_of_cuda_memory(self) -> None:
        """Test cuda OOM error detection."""
        cuda_oom_error_1 = RuntimeError("CUDA out of memory. Tried to allocate ...")
        self.assertTrue(is_out_of_cuda_memory(cuda_oom_error_1))
        cuda_oom_error_2 = RuntimeError(
            "RuntimeError: cuda runtime error (2) : out of memory"
        )
        self.assertTrue(is_out_of_cuda_memory(cuda_oom_error_2))
        not_cuda_oom_error = RuntimeError("RuntimeError: blah")
        self.assertFalse(is_out_of_cuda_memory(not_cuda_oom_error))

    def test_is_out_of_memory_error(self) -> None:
        """Test general OOM error detection."""
        cpu_oom_error = RuntimeError("DefaultCPUAllocator: can't allocate memory")
        self.assertTrue(is_out_of_memory_error(cpu_oom_error))
        cuda_oom_error_1 = RuntimeError("CUDA out of memory. Tried to allocate ...")
        self.assertTrue(is_out_of_memory_error(cuda_oom_error_1))
        cuda_oom_error_2 = RuntimeError(
            "RuntimeError: cuda runtime error (2) : out of memory"
        )
        self.assertTrue(is_out_of_memory_error(cuda_oom_error_2))
        not_oom_error = RuntimeError("RuntimeError: blah")
        self.assertFalse(is_out_of_memory_error(not_oom_error))

    def test_bytes_to_mb_gb(self) -> None:
        bytes_to_mb_test_cases = [
            (0, "0.0 MB"),
            (100000, "0.1 MB"),
            (1000000, "0.95 MB"),
            (1000000000, "0.93 GB"),
            (1000000000000, "931.32 GB"),
        ]
        for inp, expected in bytes_to_mb_test_cases:
            self.assertEqual(expected, _bytes_to_mb_gb(inp))
