# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from functools import partial
from typing import Dict, List, Type, Union
from unittest.mock import Mock

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTestUnit, TTrainUnit
from torchtnt.utils.event_handlers import log_interval

logger: logging.Logger = logging.getLogger(__name__)


def _has_method_override(
    method_name: str, instance: object, base_class: Type[object]
) -> bool:
    """
    Checks if a class instance overrides a specific method from a particular base class.
    """

    if isinstance(instance, Mock):
        # Special case Mocks for testing purposes
        return True

    instance_method = getattr(instance, method_name, None)
    if instance_method is None:
        return False
    base_method = getattr(base_class, method_name, None)
    # for functools wraps
    if hasattr(instance_method, "__wrapped__"):
        instance_method = instance_method.__wrapped__
    if isinstance(instance_method, partial):
        instance_method = instance_method.func
    if instance_method is None:
        return False

    return instance_method.__code__ != base_method.__code__


def _get_implemented_callback_mapping(
    callbacks: List[Callback],
) -> Dict[str, List[Callback]]:
    """
    Processes a list of callbacks and returns a dictionary mapping each hook
    to a list of Callback classes that implement that method.

    Since the base Callback class is a no-op for each of these methods, if the code is different between
    the callback instance passed in and the base class, there is logic to run.

    This upfront processing is useful to avoid timing and profiling overheads, especially those done each step
    in the corresponding training, evaluation or prediction loop.

    Within each hook, the original ordering from the Callback list is preserved.
    """
    callback_hooks = (
        "on_exception",
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
        "on_test_start",
        "on_test_epoch_start",
        "on_test_dataloader_iter_creation_start",
        "on_test_dataloader_iter_creation_end",
        "on_test_get_next_batch_start",
        "on_test_get_next_batch_end",
        "on_test_step_start",
        "on_test_step_end",
        "on_test_epoch_end",
        "on_test_end",
    )
    cb_overrides: Dict[str, List[Callback]] = {}
    for hook in callback_hooks:
        for cb in callbacks:
            if _has_method_override(hook, cb, Callback):
                if hook not in cb_overrides:
                    cb_overrides[hook] = [cb]
                else:
                    cb_overrides[hook].append(cb)
    return cb_overrides


class CallbackHandler:
    """
    A helper class to run and time callbacks in TorchTNT.
    """

    def __init__(self, callbacks: List[Callback]) -> None:
        self._callbacks: Dict[str, List[Callback]] = _get_implemented_callback_mapping(
            callbacks
        )

    def on_exception(
        self,
        state: State,
        unit: Union[TTrainUnit, TEvalUnit, TPredictUnit, TTestUnit],
        exc: BaseException,
    ) -> None:
        fn_name = "on_exception"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_exception(state, unit, exc)

    @log_interval("on_train_start", {"category": "callback_handler"})
    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        fn_name = "on_train_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_train_start(state, unit)

    def on_train_epoch_start(self, state: State, unit: TTrainUnit) -> None:
        fn_name = "on_train_epoch_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_train_epoch_start(state, unit)

    def on_train_dataloader_iter_creation_start(
        self, state: State, unit: TTrainUnit
    ) -> None:
        fn_name = "on_train_dataloader_iter_creation_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_train_dataloader_iter_creation_start(state, unit)

    def on_train_dataloader_iter_creation_end(
        self, state: State, unit: TTrainUnit
    ) -> None:
        fn_name = "on_train_dataloader_iter_creation_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_train_dataloader_iter_creation_end(state, unit)

    def on_train_get_next_batch_start(self, state: State, unit: TTrainUnit) -> None:
        fn_name = "on_train_get_next_batch_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_train_get_next_batch_start(state, unit)

    def on_train_get_next_batch_end(self, state: State, unit: TTrainUnit) -> None:
        fn_name = "on_train_get_next_batch_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_train_get_next_batch_end(state, unit)

    def on_train_step_start(self, state: State, unit: TTrainUnit) -> None:
        fn_name = "on_train_step_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_train_step_start(state, unit)

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        fn_name = "on_train_step_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_train_step_end(state, unit)

    @log_interval("on_train_epoch_end", {"category": "callback_handler"})
    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        fn_name = "on_train_epoch_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_train_epoch_end(state, unit)

    @log_interval("on_train_end", {"category": "callback_handler"})
    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        fn_name = "on_train_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_train_end(state, unit)

    def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
        fn_name = "on_eval_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_eval_start(state, unit)

    def on_eval_epoch_start(self, state: State, unit: TEvalUnit) -> None:
        fn_name = "on_eval_epoch_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_eval_epoch_start(state, unit)

    def on_eval_dataloader_iter_creation_start(
        self, state: State, unit: TEvalUnit
    ) -> None:
        fn_name = "on_eval_dataloader_iter_creation_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_eval_dataloader_iter_creation_start(state, unit)

    def on_eval_dataloader_iter_creation_end(
        self, state: State, unit: TEvalUnit
    ) -> None:
        fn_name = "on_eval_dataloader_iter_creation_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_eval_dataloader_iter_creation_end(state, unit)

    def on_eval_get_next_batch_start(self, state: State, unit: TEvalUnit) -> None:
        fn_name = "on_eval_get_next_batch_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_eval_get_next_batch_start(state, unit)

    def on_eval_get_next_batch_end(self, state: State, unit: TEvalUnit) -> None:
        fn_name = "on_eval_get_next_batch_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_eval_get_next_batch_end(state, unit)

    def on_eval_step_start(self, state: State, unit: TEvalUnit) -> None:
        fn_name = "on_eval_step_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_eval_step_start(state, unit)

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        fn_name = "on_eval_step_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_eval_step_end(state, unit)

    def on_eval_epoch_end(self, state: State, unit: TEvalUnit) -> None:
        fn_name = "on_eval_epoch_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_eval_epoch_end(state, unit)

    def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
        fn_name = "on_eval_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_eval_end(state, unit)

    def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
        fn_name = "on_predict_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_predict_start(state, unit)

    def on_predict_epoch_start(self, state: State, unit: TPredictUnit) -> None:
        fn_name = "on_predict_epoch_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_predict_epoch_start(state, unit)

    def on_predict_dataloader_iter_creation_start(
        self, state: State, unit: TPredictUnit
    ) -> None:
        fn_name = "on_predict_dataloader_iter_creation_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_predict_dataloader_iter_creation_start(state, unit)

    def on_predict_dataloader_iter_creation_end(
        self, state: State, unit: TPredictUnit
    ) -> None:
        fn_name = "on_predict_dataloader_iter_creation_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_predict_dataloader_iter_creation_end(state, unit)

    def on_predict_get_next_batch_start(self, state: State, unit: TPredictUnit) -> None:
        fn_name = "on_predict_get_next_batch_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_predict_get_next_batch_start(state, unit)

    def on_predict_get_next_batch_end(self, state: State, unit: TPredictUnit) -> None:
        fn_name = "on_predict_get_next_batch_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_predict_get_next_batch_end(state, unit)

    def on_predict_step_start(self, state: State, unit: TPredictUnit) -> None:
        fn_name = "on_predict_step_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_predict_step_start(state, unit)

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        fn_name = "on_predict_step_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_predict_step_end(state, unit)

    def on_predict_epoch_end(self, state: State, unit: TPredictUnit) -> None:
        fn_name = "on_predict_epoch_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_predict_epoch_end(state, unit)

    def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
        fn_name = "on_predict_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_predict_end(state, unit)

    def on_test_start(self, state: State, unit: TTestUnit) -> None:
        fn_name = "on_test_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_test_start(state, unit)

    def on_test_epoch_start(self, state: State, unit: TTestUnit) -> None:
        fn_name = "on_test_epoch_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_test_epoch_start(state, unit)

    def on_test_dataloader_iter_creation_start(
        self, state: State, unit: TTestUnit
    ) -> None:
        fn_name = "on_test_dataloader_iter_creation_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_test_dataloader_iter_creation_start(state, unit)

    def on_test_dataloader_iter_creation_end(
        self, state: State, unit: TTestUnit
    ) -> None:
        fn_name = "on_test_dataloader_iter_creation_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_test_dataloader_iter_creation_end(state, unit)

    def on_test_get_next_batch_start(self, state: State, unit: TTestUnit) -> None:
        fn_name = "on_test_get_next_batch_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_test_get_next_batch_start(state, unit)

    def on_test_get_next_batch_end(self, state: State, unit: TTestUnit) -> None:
        fn_name = "on_test_get_next_batch_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_test_get_next_batch_end(state, unit)

    def on_test_step_start(self, state: State, unit: TTestUnit) -> None:
        fn_name = "on_test_step_start"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_test_step_start(state, unit)

    def on_test_step_end(self, state: State, unit: TTestUnit) -> None:
        fn_name = "on_test_step_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_test_step_end(state, unit)

    def on_test_epoch_end(self, state: State, unit: TTestUnit) -> None:
        fn_name = "on_test_epoch_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_test_epoch_end(state, unit)

    def on_test_end(self, state: State, unit: TTestUnit) -> None:
        fn_name = "on_test_end"
        callbacks = self._callbacks.get(fn_name, [])
        for cb in callbacks:
            cb.on_test_end(state, unit)
