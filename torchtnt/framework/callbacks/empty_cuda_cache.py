# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import cast

import torch
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import EntryPoint, State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit


class EmptyCudaCache(Callback):
    """
    A callback that performs periodic emptying of cuda cache using `torch.cuda.empty_cache <https://pytorch.org/docs/stable/generated/torch.cuda.empty_cache.html#torch.cuda.empty_cache>_.

    On different ranks, reserved memory and fragmentation might diverge after several iterations.
    If different ranks trigger de-fragmentation (i.e. cudaFree and redo cudaMalloc later)
    at different times, there will be different stragglers in different iterations, which will
    hurt the performance and will get worse with larger clusters. To avoid this, this callback
    calls empty_cache() at the same cadence across all ranks.

    Args:
        step_interval: number of steps to run before emptying cuda cache
    """

    def __init__(self, step_interval: int) -> None:
        self._step_interval = step_interval

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        total_num_steps_completed = unit.train_progress.num_steps_completed
        if state.entry_point == EntryPoint.FIT:
            # if fitting, unit should also subclass EvalUnit
            unit_as_eval_unit = cast(TEvalUnit, unit)
            # if fitting, include the num eval steps completed in the total steps completed
            total_num_steps_completed += (
                unit_as_eval_unit.eval_progress.num_steps_completed
            )

        if total_num_steps_completed % self._step_interval == 0:
            torch.cuda.empty_cache()

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        total_num_steps_completed = unit.eval_progress.num_steps_completed
        if state.entry_point == EntryPoint.FIT:
            # if fitting, unit should also subclass TrainUnit
            unit_as_train_unit = cast(TTrainUnit, unit)
            # if fitting, include the num train steps completed in the total steps completed
            total_num_steps_completed += (
                unit_as_train_unit.train_progress.num_steps_completed
            )

        if total_num_steps_completed % self._step_interval == 0:
            torch.cuda.empty_cache()

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        if unit.predict_progress.num_steps_completed % self._step_interval == 0:
            torch.cuda.empty_cache()
