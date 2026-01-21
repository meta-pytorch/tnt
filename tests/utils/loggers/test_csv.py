#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from torchtnt.utils.loggers.csv import CSVLogger


class CSVLoggerTest(unittest.TestCase):
    def test_csv_log(self) -> None:
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir, "test.csv").as_posix()
            logger = CSVLogger(csv_path, steps_before_flushing=1)
            log_name = "asdf"
            log_value = 123.0
            log_step = 10
            logger.log(log_name, log_value, log_step)
            logger.close()

            with open(csv_path) as f:
                output = list(csv.DictReader(f))
                self.assertEqual(float(output[0][log_name]), log_value)
                self.assertEqual(int(output[0]["step"]), log_step)

    def test_csv_log_async(self) -> None:
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir, "test.csv").as_posix()
            logger = CSVLogger(csv_path, steps_before_flushing=1, async_write=True)
            log_name = "asdf"
            log_value = 123.0
            log_step = 10
            logger.log(log_name, log_value, log_step)
            logger.close()

            with open(csv_path) as f:
                output = list(csv.DictReader(f))
                self.assertEqual(float(output[0][log_name]), log_value)
                self.assertEqual(int(output[0]["step"]), log_step)

    def test_csv_log_async(self) -> None:
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir, "test.csv").as_posix()
            logger = CSVLogger(csv_path, steps_before_flushing=1, async_write=True)
            log_name = "asdf"
            log_value = 123.0
            log_step = 10
            logger.log(log_name, log_value, log_step)
            logger.close()

            with open(csv_path) as f:
                output = list(csv.DictReader(f))
                # pyre-fixme[16]: `_DictReadMapping` has no attribute `__getitem__`.
                self.assertEqual(float(output[0][log_name]), log_value)
                self.assertEqual(int(output[0]["step"]), log_step)
