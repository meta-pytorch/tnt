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
from torchtnt.framework.state import ActivePhase, EntryPoint, PhaseState, State
from torchtnt.framework.unit import TTestData, TTestUnit
from torchtnt.framework.utils import get_timing_context
from torchtnt.utils.timer import get_timer_summary, TimerProtocol
from torchtnt.utils.version import is_torch_version_geq

logger: logging.Logger = logging.getLogger(__name__)


def test(
    test_unit: TTestUnit,
    test_dataloader: Iterable[TTestData],
    *,
    max_steps_per_epoch: Optional[int] = None,
    callbacks: Optional[List[Callback]] = None,
    timer: Optional[TimerProtocol] = None,
) -> None:
    """
    The ``test`` entry point takes in a :class:`~torchtnt.framework.unit.TestUnit` object, a test dataloader (any Iterable), optional arguments to modify loop execution,
    and runs the test loop.

    Args:
        test_unit: an instance of :class:`~torchtnt.framework.unit.TestUnit` which implements `test_step`.
        test_dataloader: dataloader to be used during testing, which can be *any* iterable, including PyTorch DataLoader, DataLoader2, etc.
        max_steps_per_epoch: the max number of steps to run per epoch. None means test until the dataloader is exhausted.
        callbacks: an optional list of :class:`~torchtnt.framework.callback.Callback` s.
        timer: an optional Timer which will be used to time key events (using a Timer with CUDA synchronization may degrade performance).


    Below is an example of calling :py:func:`~torchtnt.framework.test`.

    .. code-block:: python

        from torchtnt.framework.test import test

        test_unit = MyTestUnit(module=...)
        test_dataloader = torch.utils.data.DataLoader(...)
        test(test_unit, test_dataloader, max_steps_per_epoch=20)

    Below is pseudocode of what the :py:func:`~torchtnt.framework.test` entry point does.

    .. code-block:: text

        set unit's tracked modules to eval mode
        call on_test_start on unit first and then callbacks
        while not done:
            call on_test_epoch_start on unit first and then callbacks
            try:
                call get_next_test_batch on unit
                call on_test_step_start on callbacks
                call test_step on unit
                increment step counter
                call on_test_step_end on callbacks
            except StopIteration:
                break
        increment epoch counter
        call on_test_epoch_end on unit first and then callbacks
        call on_test_end on unit first and then callbacks
    """
    _log_api_usage("test")
    callback_handler = CallbackHandler(callbacks or [])
    state = State(
        entry_point=EntryPoint.TEST,
        test_state=PhaseState(
            dataloader=test_dataloader,
            max_steps_per_epoch=max_steps_per_epoch,
        ),
        timer=timer,
    )
    try:
        with torch.no_grad():
            _test_impl(state, test_unit, callback_handler)
        logger.info("Finished test")
        if state.timer:
            logger.info(get_timer_summary(state.timer))
    except Exception as e:
        # TODO: log for diagnostics
        logger.info(
            f"Exception during test after the following progress: test progress: {test_unit.test_progress.get_progress_string()}:\n{e}"
        )
        test_unit.on_exception(state, e)
        callback_handler.on_exception(state, test_unit, e)
        raise e


def _test_impl(
    state: State,
    test_unit: TTestUnit,
    callback_handler: CallbackHandler,
) -> None:
    # input validation
    test_state = none_throws(state.test_state)

    state._active_phase = ActivePhase.TEST
    logger.info(
        f"Started test with max_steps_per_epoch={test_state.max_steps_per_epoch}"
    )

    # Set all modules to eval mode
    # access modules made available through AppStateMixin
    tracked_modules = test_unit.tracked_modules()
    prior_module_train_states = _set_module_training_mode(tracked_modules, False)

    test_unit.on_test_start(state)
    callback_handler.on_test_start(state, test_unit)

    # Conditionally run this to avoid running this multiple times
    # in the case of resuming from a checkpoint mid-epoch
    if test_unit.test_progress.num_steps_completed_in_epoch == 0:
        test_unit.on_test_epoch_start(state)
        callback_handler.on_test_epoch_start(state, test_unit)

    callback_handler.on_test_dataloader_iter_creation_start(state, test_unit)
    with get_timing_context(state, "test.iter(dataloader)"):
        data_iter = iter(test_state.dataloader)
    callback_handler.on_test_dataloader_iter_creation_end(state, test_unit)

    prev_steps_in_epoch = test_unit.test_progress.num_steps_completed_in_epoch

    stop_iteration_reached = False
    while not (
        state.should_stop
        or _is_epoch_done(
            test_unit.test_progress,
            test_state.max_steps_per_epoch,
            test_state.max_steps,
        )
    ):
        try:
            with get_timing_context(
                state, "test.next(data_iter)"
            ), test_state.iteration_timer.time("data_wait_time"):
                callback_handler.on_test_get_next_batch_start(state, test_unit)
                step_input = test_unit.get_next_test_batch(state, data_iter)
                callback_handler.on_test_get_next_batch_end(state, test_unit)

            with test_state.iteration_timer.time("test_iteration_time"):
                callback_handler.on_test_step_start(state, test_unit)
                test_state._step_output = test_unit.test_step(state, step_input)

                test_unit.test_progress.increment_step()
                callback_handler.on_test_step_end(state, test_unit)

                # clear step_output to avoid retaining extra memory
                test_state._step_output = None

            if (
                test_unit.test_progress.num_steps_completed_in_epoch
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
        entry_point = "test"
        if state.entry_point == EntryPoint.FIT:
            entry_point = "fit"
        logger.info(f"Reached end of test dataloader during {entry_point}")
    logger.info(
        f"Finished test epoch in {test_unit.test_progress.num_steps_completed_in_epoch} steps"
    )

    # Possibly warn about an empty dataloader
    any_steps_completed = (
        abs(test_unit.test_progress.num_steps_completed_in_epoch - prev_steps_in_epoch)
        > 0
    )
    if not any_steps_completed:
        logger.warning("No steps completed during test epoch!")

    # set progress counters for the next epoch
    test_unit.test_progress.increment_epoch()

    test_unit.on_test_epoch_end(state)
    callback_handler.on_test_epoch_end(state, test_unit)

    test_unit.on_test_end(state)
    callback_handler.on_test_end(state, test_unit)

    # Reset training mode for modules at the end of the epoch
    # This ensures that side-effects made by the loop are reset before
    # returning back to the user
    _reset_module_training_mode(tracked_modules, prior_module_train_states)
