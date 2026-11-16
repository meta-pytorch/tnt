# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from io import StringIO
from typing import cast

from torchtnt.utils.loggers.stdout import StdoutLogger
from torchtnt.utils.test_utils import captured_output


class StdoutLoggerTest(unittest.TestCase):
    def test_stdout_log(self) -> None:
        logger = StdoutLogger(precision=2)
        with captured_output() as (out, _):
            logger.log(step=0, name="metric_1", data=1.1)
            out = cast(StringIO, out)
            self.assertTrue(
                out.getvalue() == "\n[Step 0] metric_1=1.10",
                msg=repr(f"Actual output: {out.getvalue()}"),
            )

            logger.log(step=0, name="metric_2", data=1.2)
            self.assertTrue(
                out.getvalue() == "\n[Step 0] metric_1=1.10 metric_2=1.20",
                msg=repr(f"Actual output: {out.getvalue()}"),
            )

            logger.log(step=1, name="metric_1", data=2.1)
            self.assertTrue(
                out.getvalue()
                == "\n[Step 0] metric_1=1.10 metric_2=1.20\n[Step 1] metric_1=2.10",
                msg=repr(f"Actual output: {out.getvalue()}"),
            )

            logger.close()
            self.assertTrue(
                out.getvalue()
                == "\n[Step 0] metric_1=1.10 metric_2=1.20\n[Step 1] metric_1=2.10\n\n",
                msg=repr(f"Actual output: {out.getvalue()}"),
            )

    def test_stdout_log_dict(self) -> None:
        logger = StdoutLogger(precision=0)
        with captured_output() as (out, _):
            logger.log_dict(step=0, payload={"metric_1": 1, "metric_2": 1})
            out = cast(StringIO, out)
            self.assertTrue(
                out.getvalue() == "\n[Step 0] metric_1=1 metric_2=1",
                msg=repr(f"Actual output: {out.getvalue()}"),
            )

            logger.log_dict(step=0, payload={"metric_3": 1})
            self.assertTrue(
                out.getvalue() == "\n[Step 0] metric_1=1 metric_2=1 metric_3=1",
                msg=repr(f"Actual output: {out.getvalue()}"),
            )

            logger.log_dict(
                step=1, payload={"metric_1": 2, "metric_2": 2.2, "metric_3": 2.2344}
            )
            self.assertTrue(
                out.getvalue()
                == "\n[Step 0] metric_1=1 metric_2=1 metric_3=1\n[Step 1] metric_1=2 metric_2=2 metric_3=2",
                msg=repr(f"Actual output: {out.getvalue()}"),
            )

            logger.close()
            self.assertTrue(
                out.getvalue()
                == "\n[Step 0] metric_1=1 metric_2=1 metric_3=1\n[Step 1] metric_1=2 metric_2=2 metric_3=2\n\n",
                msg=repr(f"Actual output: {out.getvalue()}"),
            )

    def test_out_of_order_steps(self) -> None:
        logger = StdoutLogger(precision=2)
        with captured_output() as (out, _):
            logger.log(step=-1, name="metric_1", data=1.1234)
            out = cast(StringIO, out)
            self.assertTrue(
                out.getvalue() == "\n[Step -1] metric_1=1.12",
                msg=repr(f"Actual output: {out.getvalue()}"),
            )

            logger.log(step=10, name="metric_1", data=1.234)
            self.assertTrue(
                out.getvalue() == "\n[Step -1] metric_1=1.12\n[Step 10] metric_1=1.23",
                msg=repr(f"Actual output: {out.getvalue()}"),
            )

            logger.log_dict(step=2, payload={"metric_2": 2.1234, "metric_3": 3.987})
            self.assertTrue(
                out.getvalue()
                == "\n[Step -1] metric_1=1.12\n[Step 10] metric_1=1.23\n[Step 2] metric_2=2.12 metric_3=3.99",
                msg=repr(f"Actual output: {out.getvalue()}"),
            )
