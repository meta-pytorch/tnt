#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Set, Union
from unittest.mock import MagicMock

from torchtnt.framework._callback_handler import (
    _get_implemented_callback_mapping,
    CallbackHandler,
)
from torchtnt.framework.callback import Callback
from torchtnt.framework.callbacks.lambda_callback import Lambda
from torchtnt.framework.state import EntryPoint, State
from torchtnt.framework.unit import (
    TEvalUnit,
    TPredictUnit,
    TrainUnit,
    TTestUnit,
    TTrainData,
    TTrainUnit,
)
from torchtnt.utils.timer import Timer


class DummyCallback(Callback):
    def __init__(self) -> None:
        self.called_hooks: Set[str] = set()

    def on_exception(
        self,
        state: State,
        unit: Union[TTrainUnit, TEvalUnit, TPredictUnit, TTestUnit],
        exc: BaseException,
    ) -> None:
        self.called_hooks.add("on_exception")

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        self.called_hooks.add("on_train_start")

    def on_train_epoch_start(self, state: State, unit: TTrainUnit) -> None:
        self.called_hooks.add("on_train_epoch_start")

    def on_train_dataloader_iter_creation_start(
        self, state: State, unit: TTrainUnit
    ) -> None:
        self.called_hooks.add("on_train_dataloader_iter_creation_start")

    def on_train_dataloader_iter_creation_end(
        self, state: State, unit: TTrainUnit
    ) -> None:
        self.called_hooks.add("on_train_dataloader_iter_creation_end")

    def on_train_get_next_batch_start(self, state: State, unit: TTrainUnit) -> None:
        self.called_hooks.add("on_train_get_next_batch_start")

    def on_train_get_next_batch_end(self, state: State, unit: TTrainUnit) -> None:
        self.called_hooks.add("on_train_get_next_batch_end")

    def on_train_step_start(self, state: State, unit: TTrainUnit) -> None:
        self.called_hooks.add("on_train_step_start")

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        self.called_hooks.add("on_train_step_end")

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        self.called_hooks.add("on_train_epoch_end")

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        self.called_hooks.add("on_train_end")

    def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
        self.called_hooks.add("on_eval_start")

    def on_eval_epoch_start(self, state: State, unit: TEvalUnit) -> None:
        self.called_hooks.add("on_eval_epoch_start")

    def on_eval_dataloader_iter_creation_start(
        self, state: State, unit: TEvalUnit
    ) -> None:
        self.called_hooks.add("on_eval_dataloader_iter_creation_start")

    def on_eval_dataloader_iter_creation_end(
        self, state: State, unit: TEvalUnit
    ) -> None:
        self.called_hooks.add("on_eval_dataloader_iter_creation_end")

    def on_eval_get_next_batch_start(self, state: State, unit: TEvalUnit) -> None:
        self.called_hooks.add("on_eval_get_next_batch_start")

    def on_eval_get_next_batch_end(self, state: State, unit: TEvalUnit) -> None:
        self.called_hooks.add("on_eval_get_next_batch_end")

    def on_eval_step_start(self, state: State, unit: TEvalUnit) -> None:
        self.called_hooks.add("on_eval_step_start")

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        self.called_hooks.add("on_eval_step_end")

    def on_eval_epoch_end(self, state: State, unit: TEvalUnit) -> None:
        self.called_hooks.add("on_eval_epoch_end")

    def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
        self.called_hooks.add("on_eval_end")

    def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
        self.called_hooks.add("on_predict_start")

    def on_predict_dataloader_iter_creation_start(
        self, state: State, unit: TPredictUnit
    ) -> None:
        self.called_hooks.add("on_predict_dataloader_iter_creation_start")

    def on_predict_dataloader_iter_creation_end(
        self, state: State, unit: TPredictUnit
    ) -> None:
        self.called_hooks.add("on_predict_dataloader_iter_creation_end")

    def on_predict_get_next_batch_start(self, state: State, unit: TPredictUnit) -> None:
        self.called_hooks.add("on_predict_get_next_batch_start")

    def on_predict_epoch_start(self, state: State, unit: TPredictUnit) -> None:
        self.called_hooks.add("on_predict_epoch_start")

    def on_predict_get_next_batch_end(self, state: State, unit: TPredictUnit) -> None:
        self.called_hooks.add("on_predict_get_next_batch_end")

    def on_predict_step_start(self, state: State, unit: TPredictUnit) -> None:
        self.called_hooks.add("on_predict_step_start")

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        self.called_hooks.add("on_predict_step_end")

    def on_predict_epoch_end(self, state: State, unit: TPredictUnit) -> None:
        self.called_hooks.add("on_predict_epoch_end")

    def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
        self.called_hooks.add("on_predict_end")


class CallbackHandlerTest(unittest.TestCase):
    def test_callback_handler(self) -> None:
        """
        Test CallbackHandler with all of the hooks on Callback
        """
        unit = MagicMock(spec=TTrainUnit)
        timer = Timer()
        state = State(
            entry_point=EntryPoint.TRAIN,
            timer=timer,
            train_state=None,
        )
        callback = DummyCallback()
        called_hooks = callback.called_hooks
        cb_handler = CallbackHandler([callback])

        cb_handler.on_exception(state, unit, ValueError("test"))
        self.assertIn("on_exception", called_hooks)

        cb_handler.on_train_start(state, unit)
        self.assertIn("on_train_start", called_hooks)

        cb_handler.on_train_epoch_start(state, unit)
        self.assertIn("on_train_epoch_start", called_hooks)

        cb_handler.on_train_dataloader_iter_creation_start(state, unit)
        self.assertIn("on_train_dataloader_iter_creation_start", called_hooks)

        cb_handler.on_train_dataloader_iter_creation_end(state, unit)
        self.assertIn("on_train_dataloader_iter_creation_end", called_hooks)

        cb_handler.on_train_get_next_batch_start(state, unit)
        self.assertIn("on_train_get_next_batch_start", called_hooks)

        cb_handler.on_train_get_next_batch_end(state, unit)
        self.assertIn("on_train_get_next_batch_end", called_hooks)

        cb_handler.on_train_step_start(state, unit)
        self.assertIn("on_train_step_start", called_hooks)

        cb_handler.on_train_step_end(state, unit)
        self.assertIn("on_train_step_end", called_hooks)

        cb_handler.on_train_step_end(state, unit)
        self.assertIn("on_train_step_end", called_hooks)

        cb_handler.on_train_epoch_end(state, unit)
        self.assertIn("on_train_epoch_end", called_hooks)

        cb_handler.on_train_end(state, unit)
        self.assertIn("on_train_end", called_hooks)

        unit = MagicMock(spec=TEvalUnit)
        cb_handler.on_eval_start(state, unit)
        self.assertIn("on_eval_start", called_hooks)

        cb_handler.on_eval_epoch_start(state, unit)
        self.assertIn("on_eval_epoch_start", called_hooks)

        cb_handler.on_eval_dataloader_iter_creation_start(state, unit)
        self.assertIn("on_eval_dataloader_iter_creation_start", called_hooks)

        cb_handler.on_eval_dataloader_iter_creation_end(state, unit)
        self.assertIn("on_eval_dataloader_iter_creation_end", called_hooks)

        cb_handler.on_eval_get_next_batch_start(state, unit)
        self.assertIn("on_eval_get_next_batch_start", called_hooks)

        cb_handler.on_eval_get_next_batch_end(state, unit)
        self.assertIn("on_eval_get_next_batch_end", called_hooks)

        cb_handler.on_eval_step_start(state, unit)
        self.assertIn("on_eval_step_start", called_hooks)

        cb_handler.on_eval_step_end(state, unit)
        self.assertIn("on_eval_step_end", called_hooks)

        cb_handler.on_eval_step_end(state, unit)
        self.assertIn("on_eval_step_end", called_hooks)

        cb_handler.on_eval_epoch_end(state, unit)
        self.assertIn("on_eval_epoch_end", called_hooks)

        cb_handler.on_eval_end(state, unit)
        self.assertIn("on_eval_end", called_hooks)

        unit = MagicMock(spec=TPredictUnit)
        cb_handler.on_predict_start(state, unit)
        self.assertIn("on_predict_start", called_hooks)

        cb_handler.on_predict_epoch_start(state, unit)
        self.assertIn("on_predict_epoch_start", called_hooks)

        cb_handler.on_predict_dataloader_iter_creation_start(state, unit)
        self.assertIn("on_predict_dataloader_iter_creation_start", called_hooks)

        cb_handler.on_predict_dataloader_iter_creation_end(state, unit)
        self.assertIn("on_predict_dataloader_iter_creation_end", called_hooks)

        cb_handler.on_predict_get_next_batch_start(state, unit)
        self.assertIn("on_predict_get_next_batch_start", called_hooks)

        cb_handler.on_predict_get_next_batch_end(state, unit)
        self.assertIn("on_predict_get_next_batch_end", called_hooks)

        cb_handler.on_predict_step_start(state, unit)
        self.assertIn("on_predict_step_start", called_hooks)

        cb_handler.on_predict_step_end(state, unit)
        self.assertIn("on_predict_step_end", called_hooks)

        cb_handler.on_predict_step_end(state, unit)
        self.assertIn("on_predict_step_end", called_hooks)

        cb_handler.on_predict_epoch_end(state, unit)
        self.assertIn("on_predict_epoch_end", called_hooks)

        cb_handler.on_predict_end(state, unit)
        self.assertIn("on_predict_end", called_hooks)

    def test_get_implemented_callback_mapping(self) -> None:
        callbacks = []
        remaining_callback_hooks = (
            "on_train_start",
            "on_train_epoch_start",
            "on_train_dataloader_iter_creation_start",
            "on_train_dataloader_iter_creation_end",
            "on_train_get_next_batch_start",
            "on_train_get_next_batch_end",
            "on_train_step_start",
            "on_train_step_end",
            "on_train_epoch_end",
            "on_train_end",
            "on_eval_start",
            "on_eval_epoch_start",
            "on_eval_dataloader_iter_creation_start",
            "on_eval_dataloader_iter_creation_end",
            "on_eval_get_next_batch_start",
            "on_eval_get_next_batch_end",
            "on_eval_step_start",
            "on_eval_step_end",
            "on_eval_epoch_end",
            "on_eval_end",
            "on_predict_start",
            "on_predict_epoch_start",
            "on_predict_dataloader_iter_creation_start",
            "on_predict_dataloader_iter_creation_end",
            "on_predict_get_next_batch_start",
            "on_predict_get_next_batch_end",
            "on_predict_step_start",
            "on_predict_step_end",
            "on_predict_epoch_end",
            "on_predict_end",
        )

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def dummy_fn(x, y):
            print("foo")

        for hook in remaining_callback_hooks:
            kwargs = {hook: dummy_fn}
            callbacks.append(Lambda(**kwargs))

        implemented_cbs = _get_implemented_callback_mapping(callbacks)
        for hook in remaining_callback_hooks:
            self.assertIn(hook, implemented_cbs)
            self.assertEqual(len(implemented_cbs[hook]), 1)
        self.assertNotIn("on_exception", implemented_cbs)

    def test_callback_ordering(self) -> None:
        # ensure that callbacks are executed in the order they are passed
        class DummyUnit(TrainUnit[None]):
            def __init__(self) -> None:
                self.on_train_start_callback_order = []
                self.on_train_end_callback_order = []

            def train_step(self, state: State, data: TTrainData) -> None:
                pass

        class FirstCallback(Callback):
            callback_name = "first_callback"

            def on_train_start(self, state: State, unit: TTrainUnit) -> None:
                unit.on_train_start_callback_order.append(self.callback_name)

            def on_train_end(self, state: State, unit: TTrainUnit) -> None:
                unit.on_train_end_callback_order.append(self.callback_name)

        class SecondCallback(Callback):
            callback_name = "second_callback"

            def on_train_start(self, state: State, unit: TTrainUnit) -> None:
                unit.on_train_start_callback_order.append(self.callback_name)

            def on_train_end(self, state: State, unit: TTrainUnit) -> None:
                unit.on_train_end_callback_order.append(self.callback_name)

        unit = DummyUnit()
        state = MagicMock(spec=State)
        first_callback = FirstCallback()
        second_callback = SecondCallback()
        callback_handler = CallbackHandler([first_callback, second_callback])

        callback_handler.on_train_start(state, unit)
        callback_handler.on_train_end(state, unit)
        self.assertEqual(
            unit.on_train_start_callback_order,
            [first_callback.callback_name, second_callback.callback_name],
        )
        self.assertEqual(
            unit.on_train_end_callback_order,
            [first_callback.callback_name, second_callback.callback_name],
        )

        self.assertEqual(
            callback_handler._callbacks,
            {
                "on_train_start": [first_callback, second_callback],
                "on_train_end": [first_callback, second_callback],
            },
        )
