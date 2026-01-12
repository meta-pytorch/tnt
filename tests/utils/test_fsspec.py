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
from typing import Any

import fsspec
from torchtnt.utils.fsspec import get_filesystem


class FsTest(unittest.TestCase):
    def _test_operations(
        self,
        fs: fsspec.AbstractFileSystem,
        directory: str,
        filename: str,
        **kwargs: Any,
    ) -> None:
        """Tests normal filsystem operations on the given directory and file.

        Args:
            fs: The filesystem to use when testing.
            directory: The directory containing the file.
            filename: The name of the file.
            kwargs: Passed to any write operations as additional arguments.
        """
        filepath = os.path.join(directory, filename)

        with fs.open(filepath, mode="w", **kwargs) as f:
            f.write("blob")
        self.assertTrue(fs.exists(filepath))
        self.assertTrue(fs.isfile(filepath))

    def test_get_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self._test_operations(
                fs=get_filesystem(temp_dir),
                directory=temp_dir,
                filename="test_fs.txt",
            )
