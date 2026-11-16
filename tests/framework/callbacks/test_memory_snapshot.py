# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import tempfile
import unittest
from unittest.mock import MagicMock, Mock

from torchtnt.framework.callbacks.memory_snapshot import MemorySnapshot
from torchtnt.framework.state import EntryPoint
from torchtnt.utils.memory_snapshot_profiler import MemorySnapshotProfiler


class TestMemorySnapshot(unittest.TestCase):
    def test_on_train_step_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_snapshot = MemorySnapshot(
                memory_snapshot_profiler=MemorySnapshotProfiler(output_dir=temp_dir),
            )
            memory_snapshot.memory_snapshot_profiler = Mock()

            mock_state, mock_unit = MagicMock(), MagicMock()
            memory_snapshot.on_train_step_end(mock_state, mock_unit)

            memory_snapshot.memory_snapshot_profiler.step.assert_called_once()

    def test_on_eval_step_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_snapshot = MemorySnapshot(
                memory_snapshot_profiler=MemorySnapshotProfiler(output_dir=temp_dir),
            )
            memory_snapshot.memory_snapshot_profiler = Mock()

            mock_state, mock_unit = MagicMock(), MagicMock()
            mock_state.entry_point = EntryPoint.EVALUATE
            memory_snapshot.on_eval_step_end(mock_state, mock_unit)

            memory_snapshot.memory_snapshot_profiler.step.assert_called_once()

    def test_on_predict_step_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_snapshot = MemorySnapshot(
                memory_snapshot_profiler=MemorySnapshotProfiler(output_dir=temp_dir),
            )
            memory_snapshot.memory_snapshot_profiler = Mock()

            mock_state, mock_unit = MagicMock(), MagicMock()
            memory_snapshot.on_predict_step_end(mock_state, mock_unit)

            memory_snapshot.memory_snapshot_profiler.step.assert_called_once()
