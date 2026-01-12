# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import torchtnt.framework.callbacks.time_limit_interrupter as time_limit_interrupter
from torchtnt.framework._test_utils import DummyTrainUnit, get_dummy_train_state
from torchtnt.framework.callbacks.time_limit_interrupter import TimeLimitInterrupter


class TimeLimitInterrupterTest(unittest.TestCase):
    def test_str_to_timedelta_conversion(self) -> None:
        tli = TimeLimitInterrupter(duration="02:10:20")
        self.assertEqual(
            tli._duration, timedelta(days=2, hours=10, minutes=20).total_seconds()
        )

        with self.assertRaisesRegex(ValueError, "Invalid duration format"):
            tli = TimeLimitInterrupter(duration="2:10:20")

        with self.assertRaisesRegex(ValueError, "Invalid duration format"):
            tli = TimeLimitInterrupter(duration="02:24:20")

        with self.assertRaisesRegex(ValueError, "Invalid duration format"):
            tli = TimeLimitInterrupter(duration="02:23:60")

    @patch("time.monotonic")
    def test_should_stop(self, mock_time_monotonic: MagicMock) -> None:
        for duration in ("00:00:42", timedelta(minutes=42)):
            tli = TimeLimitInterrupter(duration=duration)
            state = get_dummy_train_state()

            # setup start time
            mock_time_monotonic.return_value = 0
            tli.on_train_start(state, Mock())

            # check that we don't stop before duration
            mock_time_monotonic.return_value = 41 * 60
            tli._should_stop(state)
            self.assertFalse(state._should_stop)

            # check that we stop after duration
            mock_time_monotonic.return_value = 42 * 60
            tli._should_stop(state)
            self.assertTrue(state._should_stop)

    @patch(f"{time_limit_interrupter.__name__}.datetime", wraps=datetime)
    @patch("time.monotonic")
    def test_should_stop_with_timestamp_limit(
        self,
        mock_time_monotonic: MagicMock,
        mock_datetime: MagicMock,
    ) -> None:
        tli = TimeLimitInterrupter(
            duration="00:00:25", timestamp=datetime(2024, 3, 12, 15, 25, 0).astimezone()
        )
        state = get_dummy_train_state()

        mock_time_monotonic.return_value = 0
        tli.on_train_start(state, Mock())

        # Max duration not reached, timestamp limit not reached -> Should not stop
        mock_datetime.now.return_value = datetime(2024, 3, 12, 15, 0, 0)
        mock_time_monotonic.return_value = 5 * 60
        tli._should_stop(state)
        self.assertFalse(state._should_stop)

        # Max duration reached, timestamp limit not reached -> Should stop
        mock_datetime.now.return_value = datetime(2024, 3, 12, 15, 0, 0)
        mock_time_monotonic.return_value = 50 * 60
        state._should_stop = False
        tli._should_stop(state)
        self.assertTrue(state._should_stop)

        # Max duration not reached, timestamp limit reached -> Should stop
        mock_datetime.now.return_value = datetime(2024, 3, 12, 15, 25, 0)
        mock_time_monotonic.return_value = 5 * 60
        state._should_stop = False
        tli._should_stop(state)
        self.assertTrue(state._should_stop)

        # Test timestamp limit reached with a different timezone, no duration -> Should stop
        tli = TimeLimitInterrupter(
            timestamp=datetime.strptime(
                "2024-03-13 10:00:00 +0000", "%Y-%m-%d %H:%M:%S %z"
            ),
        )
        state = get_dummy_train_state()
        mock_time_monotonic.return_value = 0
        tli.on_train_start(state, Mock())
        mock_datetime.now.return_value = datetime.strptime(
            "2024-03-13 9:00:00 -0100", "%Y-%m-%d %H:%M:%S %z"
        )
        tli._should_stop(state)
        self.assertTrue(state._should_stop)

        # Test not timezone aware datetime -> Expected error
        with self.assertRaisesRegex(
            ValueError, "Invalid timestamp. Expected a timezone aware datetime object."
        ):
            tli = TimeLimitInterrupter(duration="00:00:25", timestamp=datetime.now())

    @patch(f"{time_limit_interrupter.__name__}.datetime", wraps=datetime)
    @patch("time.monotonic")
    def test_should_stop_optional_params(
        self,
        mock_time_monotonic: MagicMock,
        mock_datetime: MagicMock,
    ) -> None:
        # Test only input duration
        tli = TimeLimitInterrupter(duration="00:00:42")
        self.assertEqual(tli._duration, 42 * 60)
        self.assertIsNone(tli._timestamp)

        state = get_dummy_train_state()
        mock_time_monotonic.return_value = 0
        tli.on_train_start(state, Mock())

        mock_time_monotonic.return_value = 42 * 60
        tli._should_stop(state)
        self.assertTrue(state._should_stop)

        # Test only input timestamp
        tms = datetime(2024, 3, 12, 15, 25, 0).astimezone()
        tli = TimeLimitInterrupter(timestamp=tms)
        self.assertEqual(tli._timestamp, tms)
        self.assertIsNone(tli._duration)

        state = get_dummy_train_state()
        mock_time_monotonic.return_value = 0
        tli.on_train_start(state, Mock())

        mock_datetime.now.return_value = tms
        tli._should_stop(state)
        self.assertTrue(state._should_stop)

        # Test input both duration and timestamp
        mock_time_monotonic.return_value = 0
        tms = datetime.now().astimezone()
        tli = TimeLimitInterrupter(timestamp=tms, duration="00:00:42")
        self.assertEqual(tli._timestamp, tms)
        self.assertEqual(tli._duration, 42 * 60)

        # Test no input error
        with self.assertRaisesRegex(
            ValueError,
            "Invalid parameters. Expected at least one of duration or timestamp to be specified.",
        ):
            TimeLimitInterrupter()

        # Test empty duration i.e. not input error
        with self.assertRaisesRegex(
            ValueError,
            "Invalid parameters. Expected at least one of duration or timestamp to be specified.",
        ):
            TimeLimitInterrupter(duration="")

    def test_interval(self) -> None:
        tli = TimeLimitInterrupter(duration="00:00:42", interval="epoch")
        tli._should_stop = Mock()

        state = Mock()
        unit = DummyTrainUnit(input_dim=1)

        tli.on_train_step_end(state, unit)
        tli._should_stop.assert_not_called()

        tli.on_train_epoch_end(state, unit)
        tli._should_stop.assert_called_once()

        tli = TimeLimitInterrupter(duration="00:00:42", interval="step")
        tli._should_stop = Mock()

        tli.on_train_epoch_end(state, unit)
        tli._should_stop.assert_not_called()

        tli.on_train_step_end(state, unit)
        tli._should_stop.assert_called_once()

    def test_interval_freq(self) -> None:
        tli = TimeLimitInterrupter(
            duration="00:00:42", interval="epoch", interval_freq=3
        )
        with patch.object(tli, "_should_stop") as should_stop_mock:
            state = Mock()
            unit = DummyTrainUnit(input_dim=1)

            tli.on_train_epoch_end(state, unit)  # epoch 0
            should_stop_mock.assert_called_once()

            unit.train_progress.increment_epoch()  # epoch 1
            tli.on_train_epoch_end(state, unit)
            should_stop_mock.assert_called_once()

            unit.train_progress.increment_epoch()  # epoch 2
            tli.on_train_epoch_end(state, unit)
            should_stop_mock.assert_called_once()

            unit.train_progress.increment_epoch()  # epoch 3
            tli.on_train_epoch_end(state, unit)
            self.assertEqual(should_stop_mock.call_count, 2)
