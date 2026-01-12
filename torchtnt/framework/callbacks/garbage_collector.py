# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import gc
from typing import cast

from pyre_extensions import none_throws
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import EntryPoint, State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit
from torchtnt.utils.progress import Progress


class GarbageCollector(Callback):
    """
    A callback that performs periodic synchronous garbage collection.

    In fully-synchronous distributed training, the same program is run
    across multiple processes. These processes need to communicate with each
    other, especially to communicate gradients to update model parameters.
    The overall program execution is therefore gated by the slowest running
    process. As a result, it's important that each process takes roughly the
    same amount of time to execute its code: otherwise we run into straggler
    processes. By default, Python's automatic garbage collection can be triggered at
    different points in each of these processes, creating the possibility of
    straggler processes. This callback makes it convenient to configure
    all processes performing garbage collection at the same time in the loop.

    Synchronizing the garbage collection can lead to a performance improvement.
    The frequency of garbage collection must be tuned based on the application at hand.

    By default, this callback does **generation 1** collection every step. This can free up
    some objects to be reaped with minimal overhead compared to the full garbage collection.

    Args:
        step_interval: number of steps to run before executing a full garbage cleanup
    """

    def __init__(self, step_interval: int) -> None:
        self._step_interval = step_interval

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        gc.disable()

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        gc.collect(generation=1)

        total_num_steps_completed = unit.train_progress.num_steps_completed
        if state.entry_point == EntryPoint.FIT:
            # if fitting, include the num eval steps completed in the total steps completed
            none_throws(state.eval_state)
            # if fitting, unit should also subclass EvalUnit
            unit_as_eval_unit = cast(TEvalUnit, unit)
            total_num_steps_completed += (
                unit_as_eval_unit.eval_progress.num_steps_completed
            )

        if total_num_steps_completed % self._step_interval == 0:
            gc.collect()

        # Ensure that GC is disabled, in case GC was reenabled elsewhere
        gc.disable()

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        gc.enable()

    def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
        if state.entry_point == EntryPoint.FIT:
            # if fitting, this is already handled in on_train_start
            return
        gc.disable()

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        gc.collect(generation=1)
        total_num_steps_completed = unit.eval_progress.num_steps_completed
        if state.entry_point == EntryPoint.FIT:
            train_progress = cast(Progress, unit.train_progress)
            # if fitting, include the num train steps completed in the total steps completed
            total_num_steps_completed += train_progress.num_steps_completed

        if total_num_steps_completed % self._step_interval == 0:
            gc.collect()

    def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
        if state.entry_point == EntryPoint.FIT:
            # if fitting, this will be handled in on_train_end
            return
        gc.enable()

    def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
        gc.disable()

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        gc.collect(generation=1)
        if unit.predict_progress.num_steps_completed % self._step_interval == 0:
            gc.collect()

    def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
        gc.enable()
