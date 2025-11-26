# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import patch

from torch.utils.data import DataLoader, Dataset

from torchtnt.framework._test_utils import (
    Batch,
    DummyPredictUnit,
    DummyTrainUnit,
    get_dummy_train_state,
)
from torchtnt.framework.callbacks.empty_dataloader_detector import (
    EmptyDataloaderDetectorCallback,
)
from torchtnt.framework.train import train


class MockTrainUnit(DummyTrainUnit):
    """Mock train unit for testing that extends DummyTrainUnit with step control functionality."""

    def __init__(self) -> None:
        super().__init__(input_dim=2)  # Use a default input dimension
        self._steps_completed_in_prev_epoch = 0

    def set_steps_completed_in_prev_epoch(self, steps: int) -> None:
        """Set the number of steps completed in the previous epoch."""
        self._steps_completed_in_prev_epoch = steps
        self.train_progress._num_steps_completed_in_prev_epoch = steps


class EmptyDataloaderDetectorCallbackTest(unittest.TestCase):
    def test_init_invalid_threshold(self) -> None:
        """Test that invalid threshold values raise ValueError."""
        with self.assertRaisesRegex(ValueError, "threshold must be a positive integer"):
            EmptyDataloaderDetectorCallback(threshold=0)

        with self.assertRaisesRegex(ValueError, "threshold must be a positive integer"):
            EmptyDataloaderDetectorCallback(threshold=-1)

    def test_init_valid_threshold(self) -> None:
        """Test that valid threshold values are accepted."""
        callback = EmptyDataloaderDetectorCallback(threshold=1)
        self.assertEqual(callback._threshold, 1)

        callback = EmptyDataloaderDetectorCallback(threshold=5)
        self.assertEqual(callback._threshold, 5)

    def test_train_empty_epoch_detection_with_exception(self) -> None:
        """Test that consecutive empty train epochs trigger exception when threshold is reached."""
        callback = EmptyDataloaderDetectorCallback(threshold=2)
        state = get_dummy_train_state()
        unit = MockTrainUnit()

        # First empty epoch - should not raise
        unit.set_steps_completed_in_prev_epoch(0)
        callback.on_train_epoch_end(state, unit)
        self.assertEqual(callback._consecutive_empty_train_epochs, 1)

        # Second empty epoch - should raise exception
        unit.set_steps_completed_in_prev_epoch(0)
        with self.assertRaisesRegex(
            RuntimeError,
            "Detected 2 consecutive empty train epochs, which exceeds the threshold of 2",
        ):
            callback.on_train_epoch_end(state, unit)

    def test_train_reset_counter_on_non_empty_epoch(self) -> None:
        """Test that consecutive empty epoch counter resets when a non-empty epoch occurs."""
        callback = EmptyDataloaderDetectorCallback(threshold=3)
        state = get_dummy_train_state()
        unit = MockTrainUnit()

        # First empty epoch
        unit.set_steps_completed_in_prev_epoch(0)
        callback.on_train_epoch_end(state, unit)
        self.assertEqual(callback._consecutive_empty_train_epochs, 1)

        # Second empty epoch
        unit.set_steps_completed_in_prev_epoch(0)
        callback.on_train_epoch_end(state, unit)
        self.assertEqual(callback._consecutive_empty_train_epochs, 2)

        # Non-empty epoch - should reset counter
        unit.set_steps_completed_in_prev_epoch(5)
        callback.on_train_epoch_end(state, unit)
        self.assertEqual(callback._consecutive_empty_train_epochs, 0)

        # Another empty epoch - counter should start from 1 again
        unit.set_steps_completed_in_prev_epoch(0)
        callback.on_train_epoch_end(state, unit)
        self.assertEqual(callback._consecutive_empty_train_epochs, 1)

    def test_threshold_one(self) -> None:
        """Test that threshold=1 triggers immediately on first empty epoch."""
        callback = EmptyDataloaderDetectorCallback(threshold=1)
        state = get_dummy_train_state()
        unit = MockTrainUnit()

        # First empty epoch should immediately trigger exception
        unit.set_steps_completed_in_prev_epoch(0)
        with self.assertRaisesRegex(
            RuntimeError,
            "Detected 1 consecutive empty train epochs, which exceeds the threshold of 1",
        ):
            callback.on_train_epoch_end(state, unit)

    def test_high_threshold(self) -> None:
        """Test that high threshold values work correctly."""
        callback = EmptyDataloaderDetectorCallback(threshold=5)
        state = get_dummy_train_state()
        unit = MockTrainUnit()

        # Four empty epochs should not trigger
        for i in range(4):
            unit.set_steps_completed_in_prev_epoch(0)
            callback.on_train_epoch_end(state, unit)
            self.assertEqual(callback._consecutive_empty_train_epochs, i + 1)

        # Fifth empty epoch should trigger exception
        unit.set_steps_completed_in_prev_epoch(0)
        with self.assertRaisesRegex(
            RuntimeError,
            "Detected 5 consecutive empty train epochs, which exceeds the threshold of 5",
        ):
            callback.on_train_epoch_end(state, unit)

    def test_warning_logged_for_each_empty_epoch(self) -> None:
        """Test that a warning is logged for each empty epoch."""
        callback = EmptyDataloaderDetectorCallback(threshold=3)
        state = get_dummy_train_state()
        unit = MockTrainUnit()

        with patch(
            "torchtnt.framework.callbacks.empty_dataloader_detector.logger"
        ) as mock_logger:
            # First empty epoch
            unit.set_steps_completed_in_prev_epoch(0)
            callback.on_train_epoch_end(state, unit)

            # Second empty epoch
            unit.set_steps_completed_in_prev_epoch(0)
            callback.on_train_epoch_end(state, unit)

            # Verify warnings were logged for each empty epoch
            self.assertEqual(mock_logger.warning.call_count, 2)
            warning_calls = mock_logger.warning.call_args_list
            self.assertTrue(
                any("Empty train epoch detected" in str(call) for call in warning_calls)
            )

    def test_non_empty_epochs_do_not_trigger_warnings(self) -> None:
        """Test that non-empty epochs do not trigger any warnings or exceptions."""
        callback = EmptyDataloaderDetectorCallback(threshold=2)
        state = get_dummy_train_state()
        unit = MockTrainUnit()

        with patch(
            "torchtnt.framework.callbacks.empty_dataloader_detector.logger"
        ) as mock_logger:
            # Multiple non-empty epochs
            for steps in [1, 5, 10, 100]:
                unit.set_steps_completed_in_prev_epoch(steps)
                callback.on_train_epoch_end(state, unit)

            # No warnings should be logged
            mock_logger.warning.assert_not_called()

            # Counter should remain at 0
            self.assertEqual(callback._consecutive_empty_train_epochs, 0)

    def test_empty_dataloader_detection_with_real_training_loop(self) -> None:
        """
        Test that simulates the real scenario from failed MAST job f762746046-pviolatingquery_cse.
        Tests EmptyDataloaderDetectorCallback with actual training loop and empty dataloaders.
        """

        class EmptyDataset(Dataset[Batch]):
            """Dataset that returns no data to simulate empty dataloader scenario."""

            def __len__(self) -> int:
                return 0

            def __getitem__(self, idx: int) -> Batch:
                raise IndexError("Empty dataset")

        callback_with_exception = EmptyDataloaderDetectorCallback(threshold=2)

        train_unit = DummyTrainUnit(input_dim=2)
        empty_dataloader = DataLoader(EmptyDataset(), batch_size=1)

        # This should raise an exception after 2 empty epochs
        with self.assertRaisesRegex(
            RuntimeError,
            "Detected 2 consecutive empty train epochs, which exceeds the threshold of 2",
        ):
            train(
                train_unit,
                empty_dataloader,
                max_epochs=50,  # Try to run 50 epochs but should fail at 2
                callbacks=[callback_with_exception],
            )

        self.assertEqual(callback_with_exception._consecutive_empty_train_epochs, 2)

    def test_predict_empty_epoch_detection(self) -> None:
        """Test that empty predict epoch immediately raises an exception."""
        callback = EmptyDataloaderDetectorCallback(threshold=2)
        state = get_dummy_train_state()
        unit = DummyPredictUnit(input_dim=2)

        # Set predict progress to 0 steps
        unit.predict_progress._num_steps_completed_in_prev_epoch = 0

        # Empty predict epoch should immediately raise exception
        with self.assertRaisesRegex(
            RuntimeError,
            "Empty predict epoch detected! Epoch completed 0 steps",
        ):
            callback.on_predict_epoch_end(state, unit)
