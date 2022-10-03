# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc

from torchtnt.runner.callback import Callback
from torchtnt.runner.state import State
from torchtnt.runner.unit import (
    EvalUnit,
    PredictUnit,
    TEvalData,
    TPredictData,
    TrainUnit,
    TTrainData,
)


class GarbageCollector(Callback):
    """
    A callback that performs periodic synchronous garbage collection

    In fully-synchronous distributed training, the same program is run
    across multiple processes. These processes need to communicate with each
    other, especially to communicate gradients to update model parameters.
    The overall program execution is therefore gated by the slowest running
    process. As a result, it's important that each process takes roughly the
    same amount of time to execute its code: otherwise we run into straggler
    processes. Python's automatic garbage collection can be triggered at
    different points in each of these processes, creating the possibility of
    straggler processes.

    Synchronizing the garbage collection can lead to a performance improvement.
    The frequency of garbage collection must be tuned based on the application at hand.

    Args:
        step_interval: number of steps to run before each collection
    """

    def __init__(self, step_interval: int) -> None:
        self._step_interval = step_interval

    def on_train_start(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        gc.disable()

    def on_train_step_end(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        if (
            state.train_state
            and (state.train_state.progress.num_steps_completed + 1)
            % self._step_interval
            == 0
        ):
            gc.collect()

    def on_train_end(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        gc.enable()

    def on_eval_start(self, state: State, unit: EvalUnit[TEvalData]) -> None:
        gc.disable()

    def on_eval_step_end(self, state: State, unit: EvalUnit[TEvalData]) -> None:
        if (
            state.eval_state
            and (state.eval_state.progress.num_steps_completed + 1)
            % self._step_interval
            == 0
        ):
            gc.collect()

    def on_eval_end(self, state: State, unit: EvalUnit[TEvalData]) -> None:
        gc.enable()

    def on_predict_start(self, state: State, unit: PredictUnit[TPredictData]) -> None:
        gc.disable()

    def on_predict_step_end(
        self, state: State, unit: PredictUnit[TPredictData]
    ) -> None:
        if (
            state.predict_state
            and (state.predict_state.progress.num_steps_completed + 1)
            % self._step_interval
            == 0
        ):
            gc.collect()

    def on_predict_end(self, state: State, unit: PredictUnit[TPredictData]) -> None:
        gc.enable()
