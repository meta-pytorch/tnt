#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import platform
from importlib.metadata import version as get_pkg_version

import torch
from packaging.version import Version


def is_windows() -> bool:
    """
    Is the current program running in the Windows operating system?
    """
    return platform.system() == "Windows"


def get_python_version() -> Version:
    """
    Get the current runtime Python version as a Version.

    Example::

        # if running in Python 3.8.0
        >>> get_python_version()
        '3.8.0'
    """
    return Version(platform.python_version())


def get_torch_version() -> Version:
    """
    Get the PyTorch version for the current runtime environment as a Version.

    Example::

        # if running PyTorch 1.12.0
        >>> get_torch_version()
        '1.12.0'
    """
    try:
        if hasattr(torch, "__version__"):
            pkg_version = Version(torch.__version__)
        else:
            # try importlib.metadata to infer version
            pkg_version = Version(get_pkg_version("torch"))
    except TypeError as e:
        raise TypeError("PyTorch version could not be detected automatically.") from e

    return pkg_version


def is_torch_version_geq(version: str) -> bool:
    return get_torch_version() >= Version(version)
