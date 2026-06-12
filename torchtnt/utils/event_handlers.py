#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import random
from contextlib import contextmanager
from functools import lru_cache
from typing import Dict, Generator, List, Optional

import importlib_metadata
from typing_extensions import Protocol, runtime_checkable

from .event import Event

logger: logging.Logger = logging.getLogger(__name__)


@runtime_checkable
class EventHandler(Protocol):
    def handle_event(self, event: Event) -> None: ...


_log_handlers: List[EventHandler] = []


@lru_cache(maxsize=None)
def get_event_handlers() -> List[EventHandler]:
    global _log_handlers

    # Registered event handlers through entry points
    eps = importlib_metadata.entry_points(group="tnt_event_handlers")
    for entry in eps:
        logger.debug(
            f"Attempting to register event handler {entry.name}: {entry.value}"
        )
        factory = entry.load()
        handler = factory()

        if not isinstance(handler, EventHandler):
            raise RuntimeError(
                f"The factory function for {({entry.value})} "
                "did not return a EventHandler object."
            )
        _log_handlers.append(handler)
    return _log_handlers


def log_event(event: Event) -> None:
    """
    Handle an event.

    Args:
        event: The event to handle.
    """

    for handler in get_event_handlers():
        handler.handle_event(event)


@contextmanager
def log_interval(
    name: str, metadata: Optional[Dict[str, str]] = None
) -> Generator[None, None, None]:
    unique_id = _generate_random_int64()
    if metadata is None:
        metadata = {}
    # pyrefly: ignore [no-matching-overload]
    metadata.update({"action": "start", "unique_id": unique_id})
    # pyrefly: ignore [bad-argument-type]
    start_event = Event(name=name, metadata=metadata)
    log_event(start_event)

    yield

    metadata["action"] = "end"
    # pyrefly: ignore [bad-argument-type]
    end_event = Event(name=name, metadata=metadata)
    log_event(end_event)


def _generate_random_int64() -> int:
    # avoid being influenced by externally set seed
    local_random = random.Random()
    return local_random.randint(0, 2**63 - 1)
