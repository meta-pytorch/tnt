# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import EntryPoint, State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit


class PyTorchProfiler(Callback):
    """
    A callback which profiles user code using `PyTorch Profiler <https://pytorch.org/docs/stable/profiler.html>`_.

    Args:
        profiler: a torch.profiler.profile context manager which will be used

    """

    def __init__(
        self,
        profiler: torch.profiler.profile,
    ) -> None:
        self.profiler: torch.profiler.profile = profiler

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        self.profiler.start()

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        self.profiler.step()

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        self.profiler.stop()

    def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
        # if in fit do nothing since the profiler was already started in on_train_start
        if state.entry_point == EntryPoint.EVALUATE:
            self.profiler.start()

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        self.profiler.step()

    def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
        # if in fit do nothing since the profiler will be stopped in on_train_end
        if state.entry_point == EntryPoint.EVALUATE:
            self.profiler.stop()

    def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
        self.profiler.start()

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        self.profiler.step()

    def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
        self.profiler.stop()
