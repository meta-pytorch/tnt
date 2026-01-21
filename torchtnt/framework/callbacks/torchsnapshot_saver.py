# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
from contextlib import contextmanager, ExitStack
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Union

import torch.distributed as dist
from torchtnt.framework.callbacks._checkpoint_utils import (
    _prepare_app_state,
    _prepare_app_state_for_checkpoint,
    _prepare_app_state_for_restore,
    _TRAIN_DL_STATE_KEY,
)
from torchtnt.framework.callbacks.base_checkpointer import BaseCheckpointer
from torchtnt.framework.callbacks.checkpointer_types import KnobOptions, RestoreOptions
from torchtnt.framework.state import State
from torchtnt.framework.unit import (
    AppStateMixin,
    TEvalUnit,
    TPredictUnit,
    TTrainData,
    TTrainUnit,
)
from torchtnt.framework.utils import get_timing_context
from torchtnt.utils.checkpoint import BestCheckpointConfig
from torchtnt.utils.optimizer import init_optim_state
from torchtnt.utils.rank_zero_log import rank_zero_info, rank_zero_warn
from torchtnt.utils.stateful import Stateful

try:
    import torchsnapshot
    from torchsnapshot.knobs import override_max_per_rank_io_concurrency
    from torchsnapshot.snapshot import PendingSnapshot, Snapshot

    _TStateful = torchsnapshot.Stateful
    _TORCHSNAPSHOT_AVAILABLE = True
except Exception:
    _TStateful = Stateful
    _TORCHSNAPSHOT_AVAILABLE = False

_RNG_STATE_KEY = "rng_state"


logger: logging.Logger = logging.getLogger(__name__)


class TorchSnapshotSaver(BaseCheckpointer):
    """
    A callback which periodically saves the application state during training using `TorchSnapshot <https://pytorch.org/torchsnapshot/>`_.

    This callback supplements the application state provided by :class:`torchtnt.unit.AppStateMixin`
    with the train progress, train dataloader (if applicable), and random number generator state.

    If used with :func:`torchtnt.framework.fit`, this class will also save the evaluation progress state.

    Checkpoints will be saved under ``dirpath/epoch_{epoch}_step_{step}`` where step is the *total* number of training and evaluation steps completed across all epochs.

    Args:
        dirpath: Parent directory to save snapshots to.
        save_every_n_train_steps: Frequency of steps with which to save snapshots during the train epoch. If None, no intra-epoch snapshots are generated.
        save_every_n_epochs: Frequency of epochs with which to save snapshots during training. If None, no end-of-epoch snapshots are generated.
        save_every_n_eval_epochs: Frequency of evaluation epochs with which to save checkpoints during training. Use this if wanting to save checkpoints after every eval epoch during fit.
        keep_last_n_checkpoints: Number of most recent checkpoints to keep. If None, all checkpoints are kept. If an excess of existing checkpoints are present, the oldest ones will be deleted to clean the difference. If best checkpoint config is enabled, this param will manage the top n checkpoints instead.
        best_checkpoint_config: Configuration for saving the best checkpoint based on a monitored metric. The metric is read off the attribute of the unit prior to checkpoint.
        process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
        async_checkpoint: Whether to perform asynchronous snapshotting. Default: ``True``.
        replicated: A glob-pattern of replicated key names that indicate which application state entries have the same state across all processes.
            For more information, see https://pytorch.org/torchsnapshot/main/api_reference.html#torchsnapshot.Snapshot.take.

            .. warning:: The replication property is safer to not set, and should only be used if really needed.
                         Things like metrics, grad_scalers, etc should not be marked as replicated as they
                         may contain different values across processes. If unsure, leave this field unset.

        storage_options: Additional keyword options for the storage plugin to use, to be passed to `torchsnapshot.Snapshot <https://pytorch.org/torchsnapshot/stable/api_reference.html#torchsnapshot.Snapshot>`_.
            See each storage plugin's documentation for customizations.
        knob_options: Additional keyword options for the snapshot knobs

    Note:
        If torch.distributed is available and default process group is initialized, the constructor will call a collective operation for rank 0 to broadcast the dirpath to all other ranks

    Note:
        If checkpointing FSDP model, you can set state_dict type calling `set_state_dict_type <https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type>`_ prior to starting training.

    Note:
        If best_checkpoint_config is enabled, the attribute must be on the unit upon checkpoint time, and must be castable to "float". This value must be maintained by the unit, and updated
        appropriately. For example, if logging validation accuracy, the unit must be responsible for maintaining the value and resetting it when the epoch ends.
    """

    metadata_fnames: List[str] = [".snapshot_metadata"]

    def __init__(
        self,
        dirpath: str,
        *,
        save_every_n_train_steps: Optional[int] = None,
        save_every_n_epochs: Optional[int] = None,
        save_every_n_eval_epochs: Optional[int] = None,
        keep_last_n_checkpoints: Optional[int] = None,
        best_checkpoint_config: Optional[BestCheckpointConfig] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        async_checkpoint: bool = True,
        replicated: Optional[List[str]] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        knob_options: Optional[KnobOptions] = None,
    ) -> None:
        _validate_snapshot_available()
        super().__init__(
            dirpath=dirpath,
            save_every_n_train_steps=save_every_n_train_steps,
            save_every_n_epochs=save_every_n_epochs,
            save_every_n_eval_epochs=save_every_n_eval_epochs,
            keep_last_n_checkpoints=keep_last_n_checkpoints,
            best_checkpoint_config=best_checkpoint_config,
            process_group=process_group,
        )
        self._async_checkpoint = async_checkpoint

        self._replicated: Set[str] = set(replicated or [])

        self._prev_snapshot: Optional[PendingSnapshot] = None
        self._storage_options = storage_options
        self._knob_options: KnobOptions = knob_options or KnobOptions()

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        """Validate there's no key collision for the app state."""
        app_state = _prepare_app_state(unit)
        _check_app_state_collision(app_state)

        super().on_train_start(state, unit)

    def on_exception(
        self,
        state: State,
        unit: Union[TTrainUnit, TEvalUnit, TPredictUnit],
        exc: BaseException,
    ) -> None:
        self._wait()

    def _checkpoint_impl(
        self,
        state: State,
        unit: AppStateMixin,
        *,
        checkpoint_id: str,
        hook: str,
    ) -> bool:
        """
        Checkpoint the current state of the application.
        """
        if hook not in [
            "on_train_step_end",
            "on_train_epoch_end",
            "on_train_end",
            "on_eval_epoch_end",
        ]:
            raise RuntimeError(f"Unexpected hook encountered '{hook}'")

        intra_epoch = False
        curr_snapshot_wait = False

        if hook == "on_train_step_end":
            intra_epoch = True
        elif hook == "on_train_end":
            curr_snapshot_wait = True

        app_state = _prepare_app_state_for_checkpoint(state, unit, intra_epoch)
        rng_state = torchsnapshot.RNGState()
        app_state[_RNG_STATE_KEY] = rng_state

        if self._async_checkpoint:
            with get_timing_context(
                state, f"{self.__class__.__name__}.take_async_snapshot"
            ):
                # TODO checkpoint is not truly successful
                # since this is async checkpointed, so in
                # future, add logic to set  successful flag
                # only when checkpoint is fully written
                checkpoint_success = self._async_snapshot(checkpoint_id, app_state)
                if curr_snapshot_wait:
                    self._wait()
        else:
            with get_timing_context(state, f"{self.__class__.__name__}.take_snapshot"):
                checkpoint_success = self._sync_snapshot(checkpoint_id, app_state)
        return checkpoint_success

    def _wait(self) -> None:
        if self._prev_snapshot is not None:
            self._prev_snapshot.wait()

    def _async_snapshot(
        self,
        snapshot_path: str,
        app_state: Dict[str, _TStateful],
    ) -> bool:
        prev_snapshot = self._prev_snapshot
        if prev_snapshot is not None:
            if prev_snapshot.path == snapshot_path:
                # Snapshot for this step already has been saved.
                # This can happen if we call _async_snapshot twice at the same step.
                return False
            still_pending = not prev_snapshot.done()
            if still_pending:
                rank_zero_warn(
                    (
                        "Still writing previous snapshot; waiting for it to finish before writing a new one. "
                        f"Consider increasing 'frequency' (current {self._save_every_n_train_steps})"
                    ),
                    logger=logger,
                )
                prev_snapshot.wait()

        replicated = self._replicated
        if self._replicated == {"**"}:
            replicated = _exclude_progress_from_replicated(app_state)

        replicated = self._replicated
        if self._replicated == {"**"}:
            replicated = _exclude_progress_from_replicated(app_state)

        with _override_knobs(self._knob_options):
            self._prev_snapshot = Snapshot.async_take(
                str(snapshot_path),
                app_state=app_state,
                pg=self._process_group,
                replicated=list(replicated),
                storage_options=self._storage_options,
            )
        rank_zero_info(f"Saving snapshot to path: {snapshot_path}", logger=logger)
        return True

    def _sync_snapshot(
        self,
        snapshot_path: str,
        app_state: Dict[str, _TStateful],
    ) -> bool:
        replicated = self._replicated
        if self._replicated == {"**"}:
            replicated = _exclude_progress_from_replicated(app_state)

        with _override_knobs(self._knob_options):
            rank_zero_info(
                f"Started saving snapshot to path: {snapshot_path}", logger=logger
            )
            Snapshot.take(
                str(snapshot_path),
                app_state=app_state,
                pg=self._process_group,
                replicated=list(replicated),
                storage_options=self._storage_options,
            )
        rank_zero_info(
            f"Finished saving snapshot to path: {snapshot_path}", logger=logger
        )
        return True

    @staticmethod
    def restore(
        path: str,
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        knob_options: Optional[KnobOptions] = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> None:
        """Utility method to restore snapshot state from a path.

        There are additional flags offered should the user want to skip loading the train and eval progress.
        By default, the train and eval progress are restored, if applicable.

        Args:
            path: Path of the snapshot to restore.
            unit: An instance of :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit` containing states to restore.
            train_dataloader: An optional train dataloader to restore. Note that restoring from predict or evaluate dataloaders is not supported for TorchSnapshotSaver.
            process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
            restore_options: Controls what to  filter when restoring the state.
            storage_options: Additional keyword options for the storage plugin to use, to be passed to `torchsnapshot.Snapshot <https://pytorch.org/torchsnapshot/stable/api_reference.html#torchsnapshot.Snapshot>`_. See each storage plugin's documentation for customizations.
            knob_options: Additional keyword options for the snapshot knobs
            strict: If ``False``, allows loading a snapshot even if not all keys exist in the unit's app_state.
        """

        _validate_snapshot_available()

        # initialize optimizer state skeletons for in-place loading of optimizer state with torchsnapshot
        for optimizer in unit.tracked_optimizers().values():
            init_optim_state(optimizer)

        restore_options = restore_options or RestoreOptions()
        app_state = _prepare_app_state_for_restore(unit, restore_options)
        _check_app_state_collision(app_state)

        snapshot = torchsnapshot.Snapshot(
            path, pg=process_group, storage_options=storage_options
        )

        rng_state = torchsnapshot.RNGState()
        app_state[_RNG_STATE_KEY] = rng_state

        if train_dataloader is not None:
            if not isinstance(train_dataloader, _TStateful):
                rank_zero_warn(
                    "train_dataloader was passed to `restore` but the dataloader does not implement the Stateful protocol to load states"
                )
            else:
                # request to restore the dataloader state only if
                # the persisted snapshot state includes the dataloader entry
                manifest = snapshot.get_manifest()
                for key in manifest:
                    if _TRAIN_DL_STATE_KEY in key:
                        app_state[_TRAIN_DL_STATE_KEY] = train_dataloader
                        break

                if _TRAIN_DL_STATE_KEY not in app_state:
                    rank_zero_warn(
                        "train_dataloader was passed to `restore` but no train dataloader exists in the Snapshot"
                    )

        if not strict:
            # if app_state keys not in torchsnapshot checkpoint,
            # remove them from app_state prior to checkpoint load
            missing_stateful_keys = []
            manifest = snapshot.get_manifest()
            for stateful_key in app_state:
                found = any((f"/{stateful_key}/" in key for key in manifest.keys()))
                if not found:
                    missing_stateful_keys.append(stateful_key)

            for key in missing_stateful_keys:
                rank_zero_warn(
                    f"{key} was passed to `restore` but does not exists in the snapshot"
                )
                app_state.pop(key)

        knob_options = knob_options or KnobOptions()
        with _override_knobs(knob_options):
            strict = strict or restore_options.strict
            snapshot.restore(app_state, strict=restore_options.strict)
        rank_zero_info(f"Restored snapshot from path: {path}", logger=logger)


def _exclude_progress_from_replicated(app_state: Dict[str, _TStateful]) -> Set[str]:
    """
    Excludes progress state from being replicated. Called if replicated=["**"] is passed in.
    Works by populating replicated with all possible keys from app_state, except for
    the keys that match the "{train,eval,predict}_progress/**" pattern.
    """

    filtered_replicated = set()
    progress_keys = {"train_progress", "eval_progress", "predict_progress"}
    for key in app_state.keys():
        if key in progress_keys:
            continue
        filtered_replicated.add(f"{key}/**")
    return filtered_replicated


def _validate_snapshot_available() -> None:
    if not _TORCHSNAPSHOT_AVAILABLE:
        raise RuntimeError(
            "TorchSnapshotSaver support requires torchsnapshot. "
            "Please make sure ``torchsnapshot`` is installed. "
            "Installation: https://github.com/pytorch/torchsnapshot#install"
        )


def _check_app_state_collision(app_state: Dict[str, _TStateful]) -> None:
    keys_to_check = (
        _TRAIN_DL_STATE_KEY,
        _RNG_STATE_KEY,
    )
    for key in keys_to_check:
        if key in app_state:
            raise RuntimeError(
                f"Detected collision for key in app state: {key}. TorchSnapshotSaver expects to save and load this key."
            )


@contextmanager
def _override_knobs(
    knob_options: KnobOptions,
) -> Generator[None, None, None]:
    knobs = []
    if knob_options.max_per_rank_io_concurrency:
        knobs.append(
            override_max_per_rank_io_concurrency(
                knob_options.max_per_rank_io_concurrency
            ),
        )

    with ExitStack() as stack:
        for mgr in knobs:
            stack.enter_context(mgr)
        yield
