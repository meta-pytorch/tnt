# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from contextlib import contextmanager, nullcontext
from typing import Any, ContextManager, Generator, Tuple, TypeVar

from torch.profiler import record_function
from torchtnt.framework.state import ActivePhase, State

_logger: logging.Logger = logging.getLogger(__name__)
T = TypeVar("T")


@contextmanager
def get_timing_context(
    state: State,
    event_name: str,
    # pyre-fixme[24]: Generic type `ContextManager` expects 1 type parameter.
) -> Generator[Tuple[ContextManager, ContextManager], None, None]:
    """
    Returns a context manager that records an event to a :class:`~torchtnt.utils.timer.Timer` and to PyTorch Profiler.

    Args:
        state: an instance of :class:`~torchtnt.framework.state.State`
        event_name: string identifier to use for timing
    """
    timer_context = (
        state.timer.time(event_name) if state.timer is not None else nullcontext()
    )
    profiler_context = record_function(event_name)
    with timer_context, profiler_context:
        yield (timer_context, profiler_context)


def get_current_step(unit: Any, phase: ActivePhase) -> int:  # type: ignore
    """
    For an AutoUnit, gets 0-indexed current step for train, eval, and predict phases.

    Args:
        unit: an instance of :class:`~torchtnt.framework.auto_unit.AutoUnit`
        phase: phase for which to retrieve the current step
    """
    if phase == ActivePhase.TRAIN:
        return unit.train_progress.num_steps_completed
    elif phase == ActivePhase.EVALUATE:
        return unit.eval_progress.num_steps_completed
    elif phase == ActivePhase.PREDICT:
        return unit.predict_progress.num_steps_completed
    else:
        raise ValueError(f"Unknown phase: {phase}")
