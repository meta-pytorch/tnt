# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import cast, Iterable, List, Optional

import torch
from pyre_extensions import none_throws
from torchtnt.framework._callback_handler import CallbackHandler
from torchtnt.framework._loop_utils import (
    _is_done,
    _is_epoch_done,
    _log_api_usage,
    _maybe_set_distributed_sampler_epoch,
    _reason_epoch_completed,
    _reset_module_training_mode,
    _set_module_training_mode,
)
from torchtnt.framework.callback import Callback
from torchtnt.framework.evaluate import _evaluate_impl
from torchtnt.framework.state import ActivePhase, EntryPoint, PhaseState, State
from torchtnt.framework.unit import EvalUnit, TTrainData, TTrainUnit
from torchtnt.framework.utils import get_timing_context
from torchtnt.utils.timer import get_timer_summary, TimerProtocol
from torchtnt.utils.version import is_torch_version_geq

logger: logging.Logger = logging.getLogger(__name__)


@torch.enable_grad()
def train(
    train_unit: TTrainUnit,
    train_dataloader: Iterable[TTrainData],
    *,
    max_epochs: Optional[int] = None,
    max_steps: Optional[int] = None,
    max_steps_per_epoch: Optional[int] = None,
    callbacks: Optional[List[Callback]] = None,
    timer: Optional[TimerProtocol] = None,
) -> None:
    """
    The ``train`` entry point takes in a :class:`~torchtnt.framework.unit.TrainUnit` object, a train dataloader (any Iterable), optional arguments to modify loop execution,
    and runs the training loop.

    Args:
        train_unit: an instance of :class:`~torchtnt.framework.unit.TrainUnit` which implements `train_step`.
        train_dataloader: dataloader to be used during training, which can be *any* iterable, including PyTorch DataLoader, DataLoader2, etc.
        max_epochs: the max number of epochs to run. ``None`` means no limit (infinite training) unless stopped by max_steps.
        max_steps: the max number of steps to run. ``None`` means no limit (infinite training) unless stopped by max_epochs.
        max_steps_per_epoch: the max number of steps to run per epoch. None means train until the dataloader is exhausted.
        callbacks: an optional list of :class:`~torchtnt.framework.callback.Callback` s.
        timer: an optional Timer which will be used to time key events (using a Timer with CUDA synchronization may degrade performance).


    Below is an example of calling :py:func:`~torchtnt.framework.train`.

    .. code-block:: python

        from torchtnt.framework.train import train

        train_unit = MyTrainUnit(module=..., optimizer=..., lr_scheduler=...)
        train_dataloader = torch.utils.data.DataLoader(...)
        train(train_unit, train_dataloader, max_epochs=4)

    Below is pseudocode of what the :py:func:`~torchtnt.framework.train` entry point does.

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
                except StopIteration:
                    break
            increment epoch counter
            call on_train_epoch_end on unit first and then callbacks
        call on_train_end on unit first and then callbacks
    """
    _log_api_usage("train")
    callback_handler = CallbackHandler(callbacks or [])
    state = State(
        entry_point=EntryPoint.TRAIN,
        train_state=PhaseState(
            dataloader=train_dataloader,
            max_epochs=max_epochs,
            max_steps=max_steps,
            max_steps_per_epoch=max_steps_per_epoch,
        ),
        timer=timer,
    )
    try:
        _train_impl(state, train_unit, callback_handler)
        logger.info("Finished train")
        if state.timer:
            logger.info(get_timer_summary(state.timer))
    except Exception as e:
        # TODO: log for diagnostics
        logger.info(
            f"Exception during train after the following progress: {train_unit.train_progress.get_progress_string()}:\n{e}"
        )
        train_unit.on_exception(state, e)
        callback_handler.on_exception(state, train_unit, e)
        raise e


# Enabling grad in case this function is called directly from elsewhere in the framework.
@torch.enable_grad()
def _train_impl(
    state: State,
    train_unit: TTrainUnit,
    callback_handler: CallbackHandler,
) -> None:
    train_state = none_throws(state.train_state)

    logger.info(
        f"Started train with max_epochs={train_state.max_epochs}, max_steps={train_state.max_steps}, max_steps_per_epoch={train_state.max_steps_per_epoch}"
    )
    if train_state.max_epochs is None and train_state.max_steps is None:
        logger.warning(
            "Will run infinite training, since both max_epochs and max_steps were not set."
        )
    state._active_phase = ActivePhase.TRAIN

    # Set all modules to train() mode
    # access modules made available through AppStateMixin
    tracked_modules = train_unit.tracked_modules()
    prior_module_train_states = _set_module_training_mode(tracked_modules, True)

    train_unit.on_train_start(state)
    callback_handler.on_train_start(state, train_unit)

    _maybe_run_pending_fit_eval(state, train_unit, callback_handler)

    while not (
        state.should_stop
        or _is_done(
            train_unit.train_progress, train_state.max_epochs, train_state.max_steps
        )
    ):
        _train_epoch_impl(state, train_unit, callback_handler)
        logger.info(
            "After train epoch, train progress: "
            f"num_epochs_completed = {train_unit.train_progress.num_epochs_completed}, "
            f"num_steps_completed = {train_unit.train_progress.num_steps_completed}"
        )

    train_unit.on_train_end(state)
    callback_handler.on_train_end(state, train_unit)

    # Reset training mode for modules at the end of the epoch
    # This ensures that side-effects made by the loop are reset before
    # returning back to the user
    _reset_module_training_mode(tracked_modules, prior_module_train_states)


def _maybe_run_pending_fit_eval(
    state: State,
    train_unit: TTrainUnit,
    callback_handler: CallbackHandler,
) -> None:
    if (
        state.should_stop
        or state.entry_point != EntryPoint.FIT
        or state.eval_state is None
    ):
        return
    eval_unit = cast(EvalUnit[object], train_unit)
    if not eval_unit.eval_progress.eval_pending:
        return
    logger.info("Detected pending eval from a previous fit attempt; running eval.")
    _run_fit_eval(state, train_unit, callback_handler)


def _run_fit_eval(
    state: State,
    train_unit: TTrainUnit,
    callback_handler: CallbackHandler,
) -> None:
    eval_unit = cast(EvalUnit[object], train_unit)
    eval_unit.eval_progress.mark_eval_pending()
    _evaluate_impl(
        state,
        eval_unit,
        callback_handler,
    )
    eval_unit.eval_progress.mark_eval_completed()
    state._active_phase = ActivePhase.TRAIN


def _train_epoch_impl(
    state: State,
    train_unit: TTrainUnit,
    callback_handler: CallbackHandler,
) -> None:
    logger.info("Started train epoch")
    state._active_phase = ActivePhase.TRAIN

    train_state = none_throws(state.train_state)

    evaluate_every_n_steps = None
    evaluate_every_n_epochs = None
    if state.eval_state:
        if state.eval_state.evaluate_every_n_steps:
            evaluate_every_n_steps = state.eval_state.evaluate_every_n_steps
        if state.eval_state.evaluate_every_n_epochs:
            evaluate_every_n_epochs = state.eval_state.evaluate_every_n_epochs

    # Check the progress to conditionally run this
    # to avoid running this multiple times
    # in the case of resuming from a checkpoint mid-epoch
    if train_unit.train_progress.num_steps_completed_in_epoch == 0:
        train_unit.on_train_epoch_start(state)
        callback_handler.on_train_epoch_start(state, train_unit)

    _maybe_set_distributed_sampler_epoch(
        train_state.dataloader, train_unit.train_progress.num_epochs_completed
    )

    callback_handler.on_train_dataloader_iter_creation_start(state, train_unit)
    with get_timing_context(state, "train.iter(dataloader)"):
        data_iter = iter(train_state.dataloader)
    callback_handler.on_train_dataloader_iter_creation_end(state, train_unit)

    prev_steps_in_epoch = train_unit.train_progress.num_steps_completed_in_epoch

    stop_iteration_reached = False
    while not (
        state.should_stop
        or _is_epoch_done(
            train_unit.train_progress,
            train_state.max_steps_per_epoch,
            train_state.max_steps,
        )
    ):
        try:
            with train_state.iteration_timer.time("train_and_data_iteration_time"):
                with get_timing_context(
                    state, "train.next(data_iter)"
                ), train_state.iteration_timer.time("data_wait_time"):
                    callback_handler.on_train_get_next_batch_start(state, train_unit)
                    step_input = train_unit.get_next_train_batch(state, data_iter)
                    callback_handler.on_train_get_next_batch_end(state, train_unit)

                with train_state.iteration_timer.time("train_iteration_time"):
                    callback_handler.on_train_step_start(state, train_unit)
                    train_state._step_output = train_unit.train_step(state, step_input)
                    train_unit.train_progress.increment_step()
                    callback_handler.on_train_step_end(state, train_unit)

                    # clear step_output to avoid retaining extra memory
                    train_state._step_output = None

            if (
                train_unit.train_progress.num_steps_completed_in_epoch
                - prev_steps_in_epoch
                == 5
            ):
                # Set the trainer thread name to improve debuggability. We do it after
                # 5 iterations to make sure that all the processes or thread pools
                # spawned / forked from the current process have already been created
                # and the trainer_main characterizes only the CPU thread that runs the
                # forward pass and schedules GPU work.
                if is_torch_version_geq("2.5.0"):
                    if torch.multiprocessing._get_thread_name() != "trainer_main":
                        torch.multiprocessing._set_thread_name("trainer_main")

            if (
                evaluate_every_n_steps
                and train_unit.train_progress.num_steps_completed
                % evaluate_every_n_steps
                == 0
            ):
                _run_fit_eval(state, train_unit, callback_handler)
                logger.info("Finished evaluation. Resuming training epoch")

        except StopIteration:
            stop_iteration_reached = True
            break

    epoch_end_reason = _reason_epoch_completed(
        train_unit.train_progress,
        train_state.max_steps_per_epoch,
        train_state.max_steps,
        stop_iteration_reached,
    )
    logger.info(epoch_end_reason)

    # Possibly warn about an empty dataloader
    any_steps_completed = (
        abs(
            train_unit.train_progress.num_steps_completed_in_epoch - prev_steps_in_epoch
        )
        > 0
    )
    if not any_steps_completed:
        logger.warning("No steps completed during train epoch!")
    else:
        # increment epoch happens after we log, hence we add 1 to the epoch counter
        logger.info(
            f"Completed {train_unit.train_progress.num_steps_completed_in_epoch} steps in train epoch {train_unit.train_progress.num_epochs_completed + 1}"
        )

    # set progress counters for the next epoch
    train_unit.train_progress.increment_epoch()

    train_unit.on_train_epoch_end(state)
    callback_handler.on_train_epoch_end(state, train_unit)

    if (
        evaluate_every_n_epochs
        and train_unit.train_progress.num_epochs_completed % evaluate_every_n_epochs
        == 0
    ):
        _run_fit_eval(state, train_unit, callback_handler)

    logger.info("Ended train epoch")
