#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock, patch

import torch
from packaging.version import Version
from torchtnt.utils import version


class TestVersion(unittest.TestCase):
    @patch("platform.system")
    def test_is_windows(self, mock_system: MagicMock) -> None:
        mock_system.return_value = "Linux"
        self.assertFalse(version.is_windows())

        mock_system.return_value = "Darwin"
        self.assertFalse(version.is_windows())

        mock_system.return_value = "Windows"
        self.assertTrue(version.is_windows())

    @patch("platform.python_version")
    def test_get_python_version(self, mock_python_version: MagicMock) -> None:
        mock_python_version.return_value = "3.8.0"
        self.assertEqual(version.get_python_version(), Version("3.8.0"))
        self.assertNotEqual(version.get_python_version(), Version("3.10.5"))

        mock_python_version.return_value = "3.10.5"
        self.assertNotEqual(version.get_python_version(), Version("3.8.0"))
        self.assertEqual(version.get_python_version(), Version("3.10.5"))

    def test_get_torch_version(self) -> None:
        with patch.object(torch, "__version__", "1.8.3"):
            self.assertEqual(version.get_torch_version(), Version("1.8.3"))
            self.assertNotEqual(version.get_torch_version(), Version("1.12.0"))

        with patch.object(torch, "__version__", "1.12.0"):
            self.assertNotEqual(version.get_torch_version(), Version("1.8.3"))
            self.assertEqual(version.get_torch_version(), Version("1.12.0"))

    def test_torch_version_comparators(self) -> None:
        with patch.object(torch, "__version__", "2.0.0a0"):
            self.assertFalse(version.is_torch_version_geq("2.1.0"))
