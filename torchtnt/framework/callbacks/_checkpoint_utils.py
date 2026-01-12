# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import Any, cast, Dict, Union

from pyre_extensions import none_throws
from torchtnt.framework.callbacks.checkpointer_types import RestoreOptions
from torchtnt.framework.state import ActivePhase, EntryPoint, State
from torchtnt.framework.unit import AppStateMixin, TEvalUnit, TPredictUnit, TTrainUnit
from torchtnt.utils.checkpoint import Phase
from torchtnt.utils.stateful import Stateful


# keys for use when checkpointing
_PHASE_DL_STATE_KEY_MAPPING: Dict[Phase, str] = {
    Phase.TRAIN: "train_dataloader",
    Phase.EVALUATE: "eval_dataloader",
    Phase.PREDICT: "predict_dataloader",
}
_TRAIN_DL_STATE_KEY = "train_dataloader"

_TRAIN_PROGRESS_STATE_KEY = "train_progress"
_EVAL_PROGRESS_STATE_KEY = "eval_progress"
_PREDICT_PROGRESS_STATE_KEY = "predict_progress"


def _get_step_phase_mapping(
    state: State, unit: Union[TTrainUnit, TEvalUnit, TPredictUnit]
) -> Dict[Phase, int]:
    """
    Returns a mapping of phase to step, depending on the entrypoint.
    For FIT, it always includes train and eval progress.
    """
    step_mapping = {}

    if state.entry_point in (EntryPoint.TRAIN, EntryPoint.FIT):
        train_unit = cast(TTrainUnit, unit)
        step_mapping[Phase.TRAIN] = train_unit.train_progress.num_steps_completed

    if state.entry_point in (EntryPoint.EVALUATE, EntryPoint.FIT):
        eval_unit = cast(TEvalUnit, unit)
        step_mapping[Phase.EVALUATE] = eval_unit.eval_progress.num_steps_completed

    if state.entry_point == EntryPoint.PREDICT:
        predict_unit = cast(TPredictUnit, unit)
        step_mapping[Phase.PREDICT] = predict_unit.predict_progress.num_steps_completed

    return step_mapping


def _get_epoch(state: State, unit: Union[TTrainUnit, TEvalUnit, TPredictUnit]) -> int:
    """
    Returns the epoch depending on the entrypoint. For FIT, it always returns the train epoch.
    """
    if state.entry_point in (EntryPoint.TRAIN, EntryPoint.FIT):
        train_unit = cast(TTrainUnit, unit)
        return train_unit.train_progress.num_epochs_completed

    elif state.entry_point == EntryPoint.PREDICT:
        predict_unit = cast(TPredictUnit, unit)
        return predict_unit.predict_progress.num_epochs_completed

    elif state.entry_point == EntryPoint.EVALUATE:
        eval_unit = cast(TEvalUnit, unit)
        return eval_unit.eval_progress.num_epochs_completed

    raise ValueError(f"Unknown entrypoint: {state.entry_point}")


def _prepare_app_state(unit: AppStateMixin) -> Dict[str, Any]:
    """Join together all of the tracked stateful entities to simplify registration of snapshottable states, deals with FSDP case"""
    app_state = unit.app_state()
    tracked_optimizers = unit._construct_tracked_optimizers()  # handles fsdp
    app_state.update(tracked_optimizers)
    return app_state


def _remove_app_state_keys(
    unit: AppStateMixin,
    app_state: Dict[str, Any],
    *,
    remove_modules: bool = False,
    remove_optimizers: bool = False,
    remove_lr_schedulers: bool = False,
) -> None:
    if remove_modules:
        # remove all module keys from app_state
        for module_keys in unit.tracked_modules().keys():
            app_state.pop(module_keys, None)

    if remove_optimizers:
        # remove all optimizer keys from app_state
        for optim_keys in unit.tracked_optimizers().keys():
            app_state.pop(optim_keys, None)

    if remove_lr_schedulers:
        # remove all lr scheduler keys from app_state
        for lr_scheduler_keys in unit.tracked_lr_schedulers().keys():
            app_state.pop(lr_scheduler_keys, None)


def _prepare_app_state_for_checkpoint(
    state: State, unit: AppStateMixin, intra_epoch: bool
) -> Dict[str, Stateful]:
    """
    Prepares the application state for checkpointing.
    """
    app_state = _prepare_app_state(unit)

    if state.entry_point in [EntryPoint.EVALUATE, EntryPoint.PREDICT]:
        # Since model parameters are fixed, remove them from checkpoint.
        _remove_app_state_keys(
            unit,
            app_state,
            remove_modules=True,
            remove_optimizers=True,
            remove_lr_schedulers=True,
        )

    if not intra_epoch:
        return app_state

    # for intra-epoch checkpointing, include dataloader state of the current phase
    active_dataloaders = {state.active_phase: state.active_phase_state().dataloader}

    # Special case for FIT where eval is executed every n steps. We also need to save
    # the train dataloader state. In this case, train epoch wouldn't be incremented yet.
    if (
        state.entry_point == EntryPoint.FIT
        and state.active_phase == ActivePhase.EVALUATE
        and cast(TTrainUnit, unit).train_progress.num_steps_completed_in_epoch != 0
    ):
        active_dataloaders[ActivePhase.TRAIN] = none_throws(
            state.train_state
        ).dataloader

    for active_phase, dl in active_dataloaders.items():
        if isinstance(dl, Stateful):
            dl_key = _PHASE_DL_STATE_KEY_MAPPING[active_phase.into_phase()]
            app_state[dl_key] = dl

    return app_state


def _prepare_app_state_for_restore(
    unit: AppStateMixin, restore_options: RestoreOptions
) -> Dict[str, Any]:
    """
    Prepares the application state for restoring from a checkpoint given a RestoreOptions.
    """
    app_state = _prepare_app_state(unit)

    restore_options = restore_options or RestoreOptions()

    if not restore_options.restore_train_progress:
        app_state.pop(_TRAIN_PROGRESS_STATE_KEY, None)

    if not restore_options.restore_eval_progress:
        app_state.pop(_EVAL_PROGRESS_STATE_KEY, None)

    if not restore_options.restore_predict_progress:
        app_state.pop(_PREDICT_PROGRESS_STATE_KEY, None)

    _remove_app_state_keys(
        unit,
        app_state,
        remove_modules=not restore_options.restore_modules,
        remove_optimizers=not restore_options.restore_optimizers,
        remove_lr_schedulers=not restore_options.restore_lr_schedulers,
    )

    return app_state
