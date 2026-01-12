# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Iterable, List, Optional

import torch
from pyre_extensions import none_throws
from torchtnt.framework._callback_handler import CallbackHandler
from torchtnt.framework._loop_utils import (
    _is_epoch_done,
    _log_api_usage,
    _reset_module_training_mode,
    _set_module_training_mode,
)
from torchtnt.framework.callback import Callback
from torchtnt.framework.callbacks.base_checkpointer import BaseCheckpointer
from torchtnt.framework.state import ActivePhase, EntryPoint, PhaseState, State
from torchtnt.framework.unit import TPredictData, TPredictUnit
from torchtnt.framework.utils import get_timing_context
from torchtnt.utils.timer import get_timer_summary, TimerProtocol
from torchtnt.utils.version import is_torch_version_geq

logger: logging.Logger = logging.getLogger(__name__)


def predict(
    predict_unit: TPredictUnit,
    predict_dataloader: Iterable[TPredictData],
    *,
    max_steps_per_epoch: Optional[int] = None,
    callbacks: Optional[List[Callback]] = None,
    timer: Optional[TimerProtocol] = None,
) -> None:
    """
    The ``predict`` entry point takes in a :class:`~torchtnt.framework.unit.PredictUnit` object, a train dataloader (any Iterable), optional arguments to modify loop execution,
    and runs the prediction loop.

    Args:
        predict_unit: an instance of :class:`~torchtnt.framework.unit.PredictUnit` which implements `predict_step`.
        predict_dataloader: dataloader to be used during prediction, which can be *any* iterable, including PyTorch DataLoader, DataLoader2, etc.
        max_steps_per_epoch: the max number of steps to run per epoch. None means predict until the dataloader is exhausted.
        callbacks: an optional list of :class:`~torchtnt.framework.callback.Callback` s.
        timer: an optional Timer which will be used to time key events (using a Timer with CUDA synchronization may degrade performance).


    Below is an example of calling :py:func:`~torchtnt.framework.predict`.

    .. code-block:: python

        from torchtnt.framework.predict import predict

        predict_unit = MyPredictUnit(module=..., optimizer=..., lr_scheduler=...)
        predict_dataloader = torch.utils.data.DataLoader(...)
        predict(predict_unit, predict_dataloader, max_steps_per_epoch=20)

    Below is pseudocode of what the :py:func:`~torchtnt.framework.predict` entry point does.

    .. code-block:: text

        set unit's tracked modules to eval mode
        call on_predict_start on unit first and then callbacks
        while not done:
            call on_predict_epoch_start on unit first and then callbacks
            try:
                call get_next_predict_batch on unit
                call on_predict_step_start on callbacks
                call predict_step on unit
                increment step counter
                call on_predict_step_end on callbacks
            except StopIteration:
                break
        increment epoch counter
        call on_predict_epoch_end on unit first and then callbacks
        call on_predict_end on unit first and then callbacks
    """
    _log_api_usage("predict")
    callbacks = callbacks or []
    callback_handler = CallbackHandler(callbacks)
    checkpoint_cb_exists = any(isinstance(cb, BaseCheckpointer) for cb in callbacks)

    state = State(
        entry_point=EntryPoint.PREDICT,
        predict_state=PhaseState(
            dataloader=predict_dataloader,
            max_steps_per_epoch=max_steps_per_epoch,
        ),
        timer=timer,
    )
    try:
        # all_gather using inference_mode with gloo backend is not supported. Since this collective
        # is necessary for checkpointing, we need to use torch.no_grad instead.
        # TODO: remove this once all_gather is supported in inference_mode.
        inference_ctx = torch.no_grad if checkpoint_cb_exists else torch.inference_mode
        with inference_ctx():
            _predict_impl(state, predict_unit, callback_handler)

        logger.info("Finished predict")
        if state.timer:
            logger.info(get_timer_summary(state.timer))
    except Exception as e:
        # TODO: log for diagnostics
        logger.info(
            f"Exception during predict after the following progress: {predict_unit.predict_progress.get_progress_string()}:\n{e}"
        )
        predict_unit.on_exception(state, e)
        callback_handler.on_exception(state, predict_unit, e)
        raise e


def _predict_impl(
    state: State,
    predict_unit: TPredictUnit,
    callback_handler: CallbackHandler,
) -> None:
    # input validation
    predict_state = none_throws(state.predict_state)

    if predict_unit.predict_progress.num_epochs_completed >= 1:
        logger.warning(
            "Predict epoch has already been completed. Skipping to avoid duplicate outputs."
        )
        return

    state._active_phase = ActivePhase.PREDICT
    logger.info(
        f"Started predict with max_steps_per_epoch={predict_state.max_steps_per_epoch}"
    )

    # Set all modules to eval mode
    # access modules made available through AppStateMixin
    tracked_modules = predict_unit.tracked_modules()
    prior_module_train_states = _set_module_training_mode(tracked_modules, False)

    with get_timing_context(
        state, f"{predict_unit.__class__.__name__}.on_predict_start"
    ):
        predict_unit.on_predict_start(state)
    callback_handler.on_predict_start(state, predict_unit)

    # Conditionally run this to avoid running this multiple times
    # in the case of resuming from a checkpoint mid-epoch
    if predict_unit.predict_progress.num_steps_completed_in_epoch == 0:
        with get_timing_context(
            state, f"{predict_unit.__class__.__name__}.on_predict_epoch_start"
        ):
            predict_unit.on_predict_epoch_start(state)
        callback_handler.on_predict_epoch_start(state, predict_unit)

    callback_handler.on_predict_dataloader_iter_creation_start(state, predict_unit)
    with get_timing_context(state, "predict.iter(dataloader)"):
        data_iter = iter(predict_state.dataloader)
    callback_handler.on_predict_dataloader_iter_creation_end(state, predict_unit)

    prev_steps_in_epoch = predict_unit.predict_progress.num_steps_completed_in_epoch

    stop_iteration_reached = False
    while not (
        state.should_stop
        or _is_epoch_done(
            predict_unit.predict_progress,
            predict_state.max_steps_per_epoch,
            predict_state.max_steps,
        )
    ):
        try:
            with get_timing_context(
                state, "predict.next(data_iter)"
            ), predict_state.iteration_timer.time("data_wait_time"):
                callback_handler.on_predict_get_next_batch_start(state, predict_unit)
                step_input = predict_unit.get_next_predict_batch(state, data_iter)
                callback_handler.on_predict_get_next_batch_end(state, predict_unit)

            with predict_state.iteration_timer.time("predict_iteration_time"):
                callback_handler.on_predict_step_start(state, predict_unit)
                predict_state._step_output = predict_unit.predict_step(
                    state, step_input
                )

                predict_unit.predict_progress.increment_step()
                callback_handler.on_predict_step_end(state, predict_unit)

                # clear step_output to avoid retaining extra memory
                predict_state._step_output = None

            if (
                predict_unit.predict_progress.num_steps_completed_in_epoch
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

        except StopIteration:
            stop_iteration_reached = True
            break

    if stop_iteration_reached:
        logger.info("Reached end of predict dataloader")
    logger.info(
        f"Finished prediction in {predict_unit.predict_progress.num_steps_completed_in_epoch} steps"
    )

    # Possibly warn about an empty dataloader
    any_steps_completed = (
        abs(
            predict_unit.predict_progress.num_steps_completed_in_epoch
            - prev_steps_in_epoch
        )
        > 0
    )
    if not any_steps_completed:
        logger.warning("No steps completed during predict epoch!")

    # set progress counters for the next epoch
    predict_unit.predict_progress.increment_epoch()

    predict_unit.on_predict_epoch_end(state)
    callback_handler.on_predict_epoch_end(state, predict_unit)

    predict_unit.on_predict_end(state)
    callback_handler.on_predict_end(state, predict_unit)

    # Reset training mode for modules at the end of the epoch
    # This ensures that side-effects made by the loop are reset before
    # returning back to the user
    _reset_module_training_mode(tracked_modules, prior_module_train_states)
