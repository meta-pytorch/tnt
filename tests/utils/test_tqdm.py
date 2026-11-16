#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

from torchtnt.utils.tqdm import create_progress_bar


class TQDMTest(unittest.TestCase):
    @patch("sys.stdout", new_callable=StringIO)
    @patch("sys.stderr", new_callable=StringIO)
    def test_tqdm_file(self, mock_stderr: MagicMock, mock_stdout: MagicMock) -> None:
        """
        Test the file argument to create_progress_bar
        """

        create_progress_bar(
            dataloader=["foo", "bar"],
            desc="foo",
            num_epochs_completed=0,
            num_steps_completed=0,
            max_steps=None,
            max_steps_per_epoch=None,
            file=None,
        )
        self.assertIn(
            "foo 0:   0%|          | 0/2 [00:00<?, ?it/s]", mock_stderr.getvalue()
        )
        # ensure nothing written to stdout
        self.assertEqual(mock_stdout.getvalue(), "")

        create_progress_bar(
            dataloader=["foo", "bar"],
            desc="foo",
            num_epochs_completed=0,
            num_steps_completed=0,
            max_steps=None,
            max_steps_per_epoch=None,
            file=sys.stdout,
        )
        self.assertIn(
            "foo 0:   0%|          | 0/2 [00:00<?, ?it/s]", mock_stdout.getvalue()
        )
