#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any

import fsspec
from fsspec.core import url_to_fs


def get_filesystem(path: str, **kwargs: Any) -> fsspec.AbstractFileSystem:
    """Returns the appropriate filesystem to use when handling the given path."""
    fs, _ = url_to_fs(path, **kwargs)
    return fs
