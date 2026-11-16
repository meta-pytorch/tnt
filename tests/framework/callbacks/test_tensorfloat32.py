# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import contextlib
import unittest
from typing import Iterator

import torch
from torchtnt.framework._test_utils import (
    DummyFitUnit,
    DummyPredictUnit,
    DummyTrainUnit,
    generate_random_dataloader,
)
from torchtnt.framework.callback import Callback
from torchtnt.framework.callbacks.tensorfloat32 import EnableTensorFloat32
from torchtnt.framework.fit import fit
from torchtnt.framework.predict import predict
from torchtnt.framework.state import State
from torchtnt.framework.train import train
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit


class _CheckTensorFloat32Enabled(Callback):
    def __init__(self, testcase: unittest.TestCase) -> None:
        self.testcase = testcase

    def assert_enabled(self) -> None:
        self.testcase.assertEqual(torch.get_float32_matmul_precision(), "high")
        self.testcase.assertTrue(torch.backends.cudnn.allow_tf32)
        self.testcase.assertTrue(torch.backends.cuda.matmul.allow_tf32)

    def on_train_step_start(self, state: State, unit: TTrainUnit) -> None:
        self.assert_enabled()

    def on_eval_step_start(self, state: State, unit: TEvalUnit) -> None:
        self.assert_enabled()

    def on_predict_step_start(self, state: State, unit: TPredictUnit) -> None:
        self.assert_enabled()


class EnableTensorFloat32Test(unittest.TestCase):
    @contextlib.contextmanager
    def check_proper_restore(self) -> Iterator[EnableTensorFloat32]:
        callback = EnableTensorFloat32()

        # Disable TensorFloat32
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

        yield callback

        # Original Values are Restored
        self.assertIsNone(callback.original_cuda_matmul)
        self.assertIsNone(callback.original_cudnn)
        self.assertIsNone(callback.original_float32_matmul_precision)

        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)

    def test_tensorfloat32_callback_train(self) -> None:
        input_dim = batch_size = max_epochs = 2
        dataset_len = 5

        unit = DummyTrainUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        with self.check_proper_restore() as callback:
            callbacks: list[Callback] = [callback, _CheckTensorFloat32Enabled(self)]
            train(unit, dataloader, max_epochs=max_epochs, callbacks=callbacks)

    def test_tensorfloat32_callback_fit(self) -> None:
        input_dim = batch_size = max_epochs = 2
        dataset_len = 5

        unit = DummyFitUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        with self.check_proper_restore() as callback:
            callbacks: list[Callback] = [callback, _CheckTensorFloat32Enabled(self)]
            fit(
                unit,
                dataloader,
                dataloader,
                max_epochs=max_epochs,
                callbacks=callbacks,
            )

    def test_tensorfloat32_callback_predict(self) -> None:
        input_dim = batch_size = 2
        dataset_len = 5

        unit = DummyPredictUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        with self.check_proper_restore() as callback:
            callbacks: list[Callback] = [callback, _CheckTensorFloat32Enabled(self)]
            predict(unit, dataloader, callbacks=callbacks)
