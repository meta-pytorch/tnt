# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from types import TracebackType
from typing import Optional, Protocol, Type

logger: logging.Logger = logging.getLogger(__name__)


class IProfiler(Protocol):
    """Protocol for profilers. Can be used as a context manager."""

    def __enter__(self) -> None:
        """Enters the context manager and starts the profiler."""
        pass

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """Exits the context manager and stops the profiler."""
        pass

    def start(self) -> None:
        """Starts the profiler."""
        pass

    def stop(self) -> None:
        """Stops the profiler."""
        pass

    def step(self) -> None:
        """Signals to the profiler that the next profiling step has started."""
        pass
