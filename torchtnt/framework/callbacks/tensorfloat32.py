# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Optional

import torch
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import EntryPoint, State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit
from torchtnt.utils.rank_zero_log import rank_zero_info

logger: logging.Logger = logging.getLogger(__name__)


class EnableTensorFloat32(Callback):
    """
    A callback that enables TensorFloat32 operations on CUDA.

    Args:
        float32_matmul_precision: precision to use for float32 matmul operations.
            See `torch.set_float32_matmul_precision` for details.
    """

    def __init__(self, float32_matmul_precision: str = "high") -> None:
        self.float32_matmul_precision = float32_matmul_precision

        self.original_float32_matmul_precision: Optional[str] = None
        self.original_cuda_matmul: Optional[bool] = None
        self.original_cudnn: Optional[bool] = None

    def _enable(self) -> None:
        rank_zero_info("Enabling TensorFloat32 operations on CUDA", logger=logger)
        assert self.original_float32_matmul_precision is None
        assert self.original_cuda_matmul is None
        assert self.original_cudnn is None

        self.original_float32_matmul_precision = torch.get_float32_matmul_precision()
        self.original_cuda_matmul = torch.backends.cuda.matmul.allow_tf32
        self.original_cudnn = torch.backends.cudnn.allow_tf32

        torch.set_float32_matmul_precision(self.float32_matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def _reset(self) -> None:
        rank_zero_info(
            "Restoring original TensorFloat32 permissions on CUDA", logger=logger
        )
        if self.original_float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(self.original_float32_matmul_precision)
            self.original_float32_matmul_precision = None

        if self.original_cuda_matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = self.original_cuda_matmul
            self.original_cuda_matmul = None

        if self.original_cudnn is not None:
            torch.backends.cudnn.allow_tf32 = self.original_cudnn
            self.original_cudnn = None

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        self._enable()

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        self._reset()

    def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
        if state.entry_point == EntryPoint.FIT:
            return  # if fitting, this is already handled in on_train_start
        self._enable()

    def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
        if state.entry_point == EntryPoint.FIT:
            return  # if fitting, this is already handled in on_train_end
        self._reset()

    def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
        self._enable()

    def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
        self._reset()
