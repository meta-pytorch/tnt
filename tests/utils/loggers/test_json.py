#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from torchtnt.utils.loggers.json import JSONLogger


class JSONLoggerTest(unittest.TestCase):
    def test_json_log(self) -> None:
        with TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir, "test.json").as_posix()
            logger = JSONLogger(json_path, steps_before_flushing=1)
            log_name = "asdf"
            log_value = 123.0
            log_step = 10
            logger.log(log_name, log_value, log_step)
            logger.close()

            with open(json_path) as f:
                d = json.load(f)
                print(d)
                self.assertTrue(len(d))
                self.assertEqual(d[0][log_name], log_value)
                self.assertEqual(d[0]["step"], log_step)
