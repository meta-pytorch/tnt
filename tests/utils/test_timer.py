#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import time
import unittest
from datetime import timedelta
from random import random
from unittest import mock

import torch.distributed as dist
from pyre_extensions import none_throws
from torchtnt.utils.distributed import spawn_multi_process
from torchtnt.utils.test_utils import skip_if_not_distributed
from torchtnt.utils.timer import (
    AggregatedTimer,
    BoundedTimer,
    FullSyncPeriodicTimer,
    get_durations_histogram,
    get_recorded_durations_table,
    get_synced_durations_histogram,
    get_timer_summary,
    log_elapsed_time,
    logger,
    Timer,
)


class TimerTest(unittest.TestCase):
    def assert_within_tolerance(
        self, expected: float, actual: float, percent_tolerance: float = 10
    ) -> None:
        """Assert that a value is correct within a percent tolerance"""
        error = abs(expected - actual)
        tolerance = expected * (percent_tolerance / 100)
        self.assertLess(error, tolerance)

    def test_timer_verbose(self) -> None:
        timer = Timer(verbose=True)
        with mock.patch.object(logger, "info") as mock_info:
            with timer.time("Testing timer"):
                time.sleep(0.2)
            mock_info.assert_called_once()
            self.assertTrue("Testing timer took" in mock_info.call_args.args[0])

    def test_timer_context_manager_size_bound(self) -> None:
        """Test that timer keeps the number of samples within bounds"""
        TEST_ACTION_STRING: str = "test action"
        UPPER_BOUND: int = 10
        LOWER_BOUND: int = 5
        timer = BoundedTimer(lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND)
        for i in range(1000):
            with timer.time(TEST_ACTION_STRING):
                pass
            if i > LOWER_BOUND:
                self.assertGreaterEqual(
                    len(timer.recorded_durations[TEST_ACTION_STRING]), LOWER_BOUND
                )
            self.assertLessEqual(
                len(timer.recorded_durations[TEST_ACTION_STRING]),
                UPPER_BOUND,
            )

    def test_timer_context_manager(self) -> None:
        """Test the context manager in the timer class"""

        # Generate 3 intervals between 0.5 and 2 seconds
        intervals = [(random() * 1.5) + 0.5 for _ in range(4)]

        # Basic test of context manager
        timer = Timer()
        with timer.time("action_1"):
            time.sleep(intervals[0])

        with timer.time("action_2"):
            time.sleep(intervals[1])

        # Make sure nested context managers work properly
        with timer.time("action_3"):
            with timer.time("action_4"):
                time.sleep(intervals[2])

        for action in ("action_1", "action_2", "action_3", "action_4"):
            self.assertIn(action, timer.recorded_durations.keys())

        self.assertLess(
            timer.recorded_durations["action_4"][0],
            timer.recorded_durations["action_3"][0],
        )
        for i in range(3):
            self.assert_within_tolerance(
                timer.recorded_durations[f"action_{i+1}"][0], intervals[i]
            )
        self.assert_within_tolerance(
            timer.recorded_durations["action_4"][0], intervals[2]
        )

    def test_get_timer_summary(self) -> None:
        """Test the get_timer_summary function"""

        timer = Timer()
        summary = get_timer_summary(timer)
        self.assertEqual(summary, f"Timer Report{os.linesep}")

        with timer.time("action_1"):
            time.sleep(0.5)
        summary = get_timer_summary(timer)
        self.assertTrue("action_1" in summary)

    def test_invalid_get_synced_durations_histogram_percentiles(self) -> None:
        with self.assertRaisesRegex(ValueError, "Percentile must be between 0 and 100"):
            get_synced_durations_histogram({}, percentiles=(-1,))
        with self.assertRaisesRegex(ValueError, "Percentile must be between 0 and 100"):
            get_synced_durations_histogram({}, percentiles=(101,))

    def test_get_synced_durations_histogram(self) -> None:
        recorded_durations = {
            "bar": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "foo": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "baz": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        }
        percentiles = (10.0, 25.0, 50.0, 75.0, 95.0, 99.0)
        durations = get_durations_histogram(recorded_durations, percentiles)
        synced_durations = get_synced_durations_histogram(
            recorded_durations, percentiles
        )
        expected_durations = {
            "bar": {
                "p10.0": 4.0,
                "p25.0": 6.0,
                "p50.0": 8.0,
                "p75.0": 10.0,
                "p95.0": 11.0,
                "p99.0": 11.0,
                "avg": 8.0,
            },
            "foo": {
                "p10.0": 1.0,
                "p25.0": 3.0,
                "p50.0": 5.0,
                "p75.0": 7.0,
                "p95.0": 9.0,
                "p99.0": 9.0,
                "avg": 5.5,
            },
            "baz": {
                "p10.0": 7.0,
                "p25.0": 9.0,
                "p50.0": 11.0,
                "p75.0": 13.0,
                "p95.0": 15.0,
                "p99.0": 15.0,
                "avg": 11.5,
            },
        }
        self.assertEqual(durations, expected_durations)
        self.assertEqual(durations, synced_durations)

    @staticmethod
    def _get_synced_durations_histogram_multi_process() -> None:
        if dist.get_rank() == 0:
            recorded_durations = {
                "foo": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "bar": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            }
        else:
            recorded_durations = {
                "foo": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                "bar": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "baz": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            }
        durations = get_synced_durations_histogram(
            recorded_durations, percentiles=(10.0, 25.0, 50.0, 75.0, 95.0, 99.0)
        )
        expected_durations = {
            "foo": {
                "p10.0": 2.0,
                "p25.0": 4.0,
                "p50.0": 7.0,
                "p75.0": 9.0,
                "p95.0": 11.0,
                "p99.0": 11.0,
                "avg": 6.684210526315789,
            },
            "bar": {
                "p10.0": 2.0,
                "p25.0": 4.0,
                "p50.0": 7.0,
                "p75.0": 9.0,
                "p95.0": 11.0,
                "p99.0": 11.0,
                "avg": 6.684210526315789,
            },
            "baz": {
                "p10.0": 7.0,
                "p25.0": 9.0,
                "p50.0": 11.0,
                "p75.0": 13.0,
                "p95.0": 15.0,
                "p99.0": 15.0,
                "avg": 11.5,
            },
        }
        tc = unittest.TestCase()
        tc.assertEqual(durations, expected_durations)

    @skip_if_not_distributed
    def test_get_synced_durations_histogram_multi_process(self) -> None:
        spawn_multi_process(
            2, "gloo", self._get_synced_durations_histogram_multi_process
        )

    def test_timer_fn(self) -> None:
        with log_elapsed_time("test"):
            pass

    def test_get_recorded_durations_table(self) -> None:
        # empty input
        empty_input = get_recorded_durations_table({})
        assert empty_input == ""

        # no recorded duration values
        no_recorded_duration_input = get_recorded_durations_table({"op": {}})
        assert no_recorded_duration_input == ""

        # valid input
        valid_input = get_recorded_durations_table({"op": {"p50": 1, "p90": 2}})
        assert (
            valid_input
            == "\n| Name   |   p50 |   p90 |\n|:-------|------:|------:|\n| op     |     1 |     2 |"
        )

    def test_aggregated_timer_statistics(self) -> None:
        """Test that AggregatedTimer maintains correct statistics"""
        timer = AggregatedTimer()

        # Time an action multiple times with known durations
        sleep_times = [1.0, 1.5, 2, 2.5, 1.0]
        for sleep_time in sleep_times:
            with timer.time("test_action"):
                time.sleep(sleep_time)

        with timer.time("fast_action"):
            time.sleep(0.05)

        # Generate report to calculate mean_duration
        report = timer._make_report()
        # test_action should be first
        test_action_report = report.timed_action_stats[0]
        fast_action_report = report.timed_action_stats[1]

        # Verify for test_action
        self.assertEqual(test_action_report.num_calls, 5)
        expected_total = sum(sleep_times)
        self.assert_within_tolerance(
            test_action_report.total_duration, expected_total, 20
        )
        self.assert_within_tolerance(
            test_action_report.mean_duration, expected_total / 5
        )

        # Verify for fast_action
        self.assertEqual(fast_action_report.num_calls, 1)
        self.assert_within_tolerance(fast_action_report.total_duration, 0.05)
        self.assertGreater(
            test_action_report.mean_duration, fast_action_report.mean_duration
        )

    def test_aggregated_timer_multiple_actions(self) -> None:
        timer = AggregatedTimer()

        actions = ["dataloader", "forward", "backward", "optimizer"]
        call_counts = [10, 10, 10, 5]

        for action, count in zip(actions, call_counts):
            for _ in range(count):
                with timer.time(action):
                    time.sleep(0.001)

        # Check that all actions are recorded
        self.assertEqual(len(timer._aggregate_stats), 4)

        for action, expected_count in zip(actions, call_counts):
            self.assertEqual(timer._aggregate_stats[action].num_calls, expected_count)

        # Check report
        report = timer._make_report()
        self.assertEqual(len(report.timed_action_stats), 4)
        self.assertEqual(report.total_calls, sum(call_counts))

        total_percentage = sum(
            stats.percentage_of_total_time for stats in report.timed_action_stats
        )
        self.assert_within_tolerance(total_percentage, 100.0, 1)


class FullSyncPeriodicTimerTest(unittest.TestCase):
    @classmethod
    def _full_sync_worker_without_timeout(
        cls,
    ) -> bool:
        process_group = dist.group.WORLD
        interval_threshold = timedelta(seconds=5)
        fsp_timer = FullSyncPeriodicTimer(
            interval_threshold, none_throws(process_group)
        )
        fsp_timer_check_val = fsp_timer.check()
        fsp_timer.wait_remaining_work()
        return fsp_timer_check_val

    @classmethod
    def _full_sync_worker_with_timeout(cls, timeout: int) -> bool:
        process_group = dist.group.WORLD
        interval_threshold = timedelta(seconds=5)
        fsp_timer = FullSyncPeriodicTimer(
            interval_threshold, none_throws(process_group)
        )
        time.sleep(timeout)
        fsp_timer.check()  # self._prev_work is assigned, next time the check is called, it will be executed
        return fsp_timer.check()  # Since 8>5, we will see flag set to True

    def test_full_sync_pt_multi_process_check_false(self) -> None:
        mp_list = spawn_multi_process(2, "gloo", self._full_sync_worker_without_timeout)
        # Both processes should return False
        self.assertEqual(mp_list, [False, False])

    def test_full_sync_pt_multi_process_check_true(self) -> None:
        mp_list = spawn_multi_process(2, "gloo", self._full_sync_worker_with_timeout, 8)
        # Both processes should return True
        self.assertEqual(mp_list, [True, True])

    def test_full_sync_pt_multi_process_edgecase(self) -> None:
        mp_list = spawn_multi_process(2, "gloo", self._full_sync_worker_with_timeout, 5)

        # Both processes should return True
        self.assertEqual(mp_list, [True, True])

        # Launch 2 worker processes. Each will check time diff >= interval threshold
        mp_list = spawn_multi_process(2, "gloo", self._full_sync_worker_with_timeout, 4)

        # Both processes should return False
        self.assertEqual(mp_list, [False, False])
