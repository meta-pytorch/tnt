# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Iterable, List, Optional

import torch
from torchtnt.framework._callback_handler import CallbackHandler
from torchtnt.framework._loop_utils import _log_api_usage
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import EntryPoint, PhaseState, State
from torchtnt.framework.test import _test_impl
from torchtnt.framework.train import _train_impl
from torchtnt.framework.unit import (
    EvalUnit,
    TestUnit,
    TEvalData,
    TrainUnit,
    TTestData,
    TTrainData,
    TTrainUnit,
)
from torchtnt.utils.timer import get_timer_summary, TimerProtocol

logger: logging.Logger = logging.getLogger(__name__)


def fit(
    unit: TTrainUnit,
    train_dataloader: Iterable[TTrainData],
    eval_dataloader: Iterable[TEvalData],
    *,
    max_epochs: Optional[int] = None,
    max_steps: Optional[int] = None,
    max_train_steps_per_epoch: Optional[int] = None,
    max_eval_steps_per_epoch: Optional[int] = None,
    evaluate_every_n_steps: Optional[int] = None,
    evaluate_every_n_epochs: Optional[int] = 1,
    callbacks: Optional[List[Callback]] = None,
    timer: Optional[TimerProtocol] = None,
    test_dataloader: Optional[Iterable[TTestData]] = None,
    max_test_steps: Optional[int] = None,
) -> None:
    """
    The ``fit`` entry point interleaves training and evaluation loops. The ``fit`` entry point takes in an object which subclasses both :class:`~torchtnt.framework.unit.TrainUnit` and :class:`~torchtnt.framework.unit.EvalUnit`, train and eval dataloaders (any Iterables), optional arguments to modify loop execution,
    and runs the fit loop. Optionally, a test dataloader can be provided to run a test phase after training completes.

    Args:
        unit: an instance that subclasses both :class:`~torchtnt.framework.unit.TrainUnit` and :class:`~torchtnt.framework.unit.EvalUnit`,
         implementing :meth:`~torchtnt.framework.TrainUnit.train_step` and :meth:`~torchtnt.framework.EvalUnit.eval_step`.
        train_dataloader: dataloader to be used during training, which can be *any* iterable, including PyTorch DataLoader, DataLoader2, etc.
        eval_dataloader: dataloader to be used during evaluation, which can be *any* iterable, including PyTorch DataLoader, DataLoader2, etc.
        max_epochs: the max number of epochs to run for training. ``None`` means no limit (infinite training) unless stopped by max_steps.
        max_steps: the max number of steps to run for training. ``None`` means no limit (infinite training) unless stopped by max_epochs.
        max_train_steps_per_epoch: the max number of steps to run per epoch for training. None means train until ``train_dataloader`` is exhausted.
        max_eval_steps_per_epoch: the max number of steps to run per epoch for evaluation. None means evaluate until ``eval_dataloader`` is exhausted.
        evaluate_every_n_steps: how often to run the evaluation loop in terms of training steps.
        evaluate_every_n_epochs: how often to run the evaluation loop in terms of training epochs.
        callbacks: an optional list of callbacks.
        timer: an optional Timer which will be used to time key events (using a Timer with CUDA synchronization may degrade performance).
        test_dataloader: an optional dataloader to be used during testing after training completes.
        max_test_steps: the max number of steps to run for testing. None means test until ``test_dataloader`` is exhausted.

    Below is an example of calling :py:func:`~torchtnt.framework.fit`.

    .. code-block:: python

        from torchtnt.framework.fit import fit

        fit_unit = MyFitUnit(module=..., optimizer=..., lr_scheduler=...)
        train_dataloader = torch.utils.data.DataLoader(...)
        eval_dataloader = torch.utils.data.DataLoader(...)
        fit(fit_unit, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, max_epochs=4)

    Below is pseudocode of what the :py:func:`~torchtnt.framework.fit` entry point does.

    .. code-block:: text

        set unit's tracked modules to train mode
        call on_train_start on unit first and then callbacks
        while training is not done:
            while epoch is not done:
                call on_train_epoch_start on unit first and then callbacks
                try:
                    call get_next_train_batch on unit
                    call on_train_step_start on callbacks
                    call train_step on unit
                    increment step counter
                    call on_train_step_end on callbacks
                    if should evaluate after this step:
                        run eval loops
                except StopIteration:
                    break
            increment epoch counter
            call on_train_epoch_end on unit first and then callbacks
            if should evaluate after this epoch:
                run eval loop
        call on_train_end on unit first and then callbacks
    """
    _log_api_usage("fit")

    # input validation
    if not isinstance(unit, TrainUnit):
        raise TypeError("Expected unit to implement TrainUnit interface.")
    if not isinstance(unit, EvalUnit):
        raise TypeError("Expected unit to implement EvalUnit interface.")
    if test_dataloader is not None and not isinstance(unit, TestUnit):
        raise TypeError(
            "Expected unit to implement TestUnit interface when test_dataloader is provided."
        )

    callback_handler = CallbackHandler(callbacks or [])
    state = State(
        entry_point=EntryPoint.FIT,
        train_state=PhaseState(
            dataloader=train_dataloader,
            max_epochs=max_epochs,
            max_steps=max_steps,
            max_steps_per_epoch=max_train_steps_per_epoch,
        ),
        eval_state=PhaseState(
            dataloader=eval_dataloader,
            max_steps_per_epoch=max_eval_steps_per_epoch,
            evaluate_every_n_steps=evaluate_every_n_steps,
            evaluate_every_n_epochs=evaluate_every_n_epochs,
        ),
        test_state=(
            PhaseState(
                dataloader=test_dataloader,
                max_steps_per_epoch=max_test_steps,
            )
            if test_dataloader is not None
            else None
        ),
        timer=timer,
    )

    logger.info(
        f"Started fit with max_epochs={max_epochs} "
        f"max_steps={max_steps} "
        f"max_train_steps_per_epoch={max_train_steps_per_epoch} "
        f"max_eval_steps_per_epoch={max_eval_steps_per_epoch} "
        f"evaluate_every_n_steps={evaluate_every_n_steps} "
        f"evaluate_every_n_epochs={evaluate_every_n_epochs} "
    )

    try:
        _train_impl(state, unit, callback_handler)

        # Run test phase after training completes, if test_dataloader was provided
        if test_dataloader is not None:
            with torch.no_grad():
                _test_impl(state, unit, callback_handler)

        logger.info("Finished fit")
        if state.timer:
            logger.info(get_timer_summary(state.timer))
    except Exception as e:
        # TODO: log for diagnostics
        logger.info(
            f"Exception during fit after the following progress: train progress: {unit.train_progress.get_progress_string()} eval progress: {unit.eval_progress.get_progress_string()}:\n{e}"
        )
        unit.on_exception(state, e)
        callback_handler.on_exception(state, unit, e)
        raise e
