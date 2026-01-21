# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import time
from concurrent.futures import Future
from typing import Any, cast, Dict, Iterable, List, Optional, Union

import torch
import torch.distributed as dist
from pyre_extensions import none_throws
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint._fsspec_filesystem import (
    FsspecReader as Reader,
    FsspecWriter as Writer,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.planner import LoadPlanner, SavePlanner
from torch.distributed.checkpoint.staging import AsyncStager
from torchtnt.framework.state import EntryPoint

try:
    from torch.distributed.checkpoint.state_dict import _init_optim_state
except ImportError:

    def noop(_: Any) -> None:
        return None

    _init_optim_state = noop

from torch.distributed.checkpoint.storage import StorageReader, StorageWriter
from torchtnt.framework.callbacks._checkpoint_utils import (
    _PHASE_DL_STATE_KEY_MAPPING,
    _prepare_app_state_for_checkpoint,
    _prepare_app_state_for_restore,
)
from torchtnt.framework.callbacks.base_checkpointer import BaseCheckpointer
from torchtnt.framework.callbacks.checkpointer_types import KnobOptions, RestoreOptions
from torchtnt.framework.state import State
from torchtnt.framework.unit import (
    AppStateMixin,
    TEvalData,
    TEvalUnit,
    TPredictData,
    TPredictUnit,
    TTrainData,
    TTrainUnit,
)
from torchtnt.framework.utils import get_timing_context
from torchtnt.utils.checkpoint import BestCheckpointConfig, CheckpointPath, Phase
from torchtnt.utils.distributed import get_or_create_gloo_pg
from torchtnt.utils.rank_zero_log import rank_zero_info, rank_zero_warn
from torchtnt.utils.stateful import MultiStateful, Stateful
from typing_extensions import TypeAlias

logger: logging.Logger = logging.getLogger(__name__)

TDataloader: TypeAlias = Union[
    Iterable[TTrainData], Iterable[TEvalData], Iterable[TPredictData]
]


class DistributedCheckpointSaver(BaseCheckpointer):
    """
    A callback which periodically saves the application state during training using `Distributed Checkpoint <https://pytorch.org/docs/stable/distributed.checkpoint.html>`_.

    This callback supplements the application state provided by :class:`torchtnt.unit.AppStateMixin`
    with the progress, and active dataloaders (if applicable).

    The keys saved in a checkpoint will vary depending on the entrypoint used.
        - :func:`torchtnt.framework.train`: Save model weights along with other Statefuls, train_progress, and train_dataloader if saving within an epoch.
        - :func:`torchtnt.framework.fit`: Save model weights along with other Statefuls, train_progress, eval_progress. In this case both train_dataloader and eval_dataloader
            may be saved depending on the `evaluate_every_n_steps` configuration.
        - :func:`torchtnt.framework.evaluate`: Save only metrics and miscellaneous Statefuls, excluding model weights since they will not change. Save eval_progress and eval_dataloader.
        - :func:`torchtnt.framework.predict`: Save only metrics and miscellaneous Statefuls, excluding model weights since they will not change. Save predict_progress and predict_dataloader.

    When restoring from a checkpoint, the following is assumed:
        - If resuming from train or fit, the module parameters are restored, along with the progress and any other statefuls.
            The train_dataloader and evaluate_dataloader are restored if they are present in the checkpoint.
        - If resuming from evaluate or predict, the module parameters are not restored. Instead, it's expected that users will load them within their units.
            In this case only the progress, metrics, and dataloaders are loaded.

    Checkpoints will be saved under ``dirpath/epoch_{epoch}_<phase>_step_{step}`` where step is the *total* number of training steps completed across all epochs.

    Args:
        dirpath: Parent directory to save snapshots to.
        save_every_n_train_steps: Frequency of steps with which to save snapshots during the train epoch. If None, no intra-epoch snapshots are generated.
        save_every_n_epochs: Frequency of epochs with which to save checkpoints during training. If None, no end-of-epoch checkpoints are generated.
        save_every_n_eval_epochs: Frequency of evaluation epochs with which to save checkpoints during training. Use this if wanting to save checkpoints after every eval epoch during fit.
        save_every_n_eval_steps: Frequency of evaluation steps with which to save checkpoints during training. Use this if wanting to save checkpoints during evaluate.
        save_every_n_predict_steps: Frequency of prediction steps with which to save checkpoints during training. Use this if wanting to save checkpoints during using predict entrypoint.
        keep_last_n_checkpoints: Number of most recent checkpoints to keep. If None, all checkpoints are kept. If an excess of existing checkpoints are present, the oldest ones will be deleted to clean the difference. If best checkpoint config is enabled, this param will manage the top n checkpoints instead.
        best_checkpoint_config: Configuration for saving the best checkpoint based on a monitored metric. The metric is read off the attribute of the unit prior to checkpoint.
        process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
        async_checkpoint: Whether to perform asynchronous checkpointing. Default: ``True``.
        knob_options: Additional keyword options for StorageWriter. <https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter/>

    Note:
        If torch.distributed is available, there should be a process group is initialized. In this case DCP assumes the intention is to save/load checkpoints in distributed fashion.
        Additionally, a gloo process group must be initialized for async_checkpoint. For workloads that require nccl, the recommended initialization is 'cpu:gloo,cuda:nccl'

    Note:
        If checkpointing FSDP model, you can set state_dict type calling `set_state_dict_type <https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type>`_ prior to starting training.

    Note:
        If best_checkpoint_config is enabled, the attribute must be on the unit upon checkpoint time, and must be castable to "float". This value must be maintained by the unit, and updated
        appropriately. For example, if logging validation accuracy, the unit must be responsible for maintaining the value and resetting it when the epoch ends.
    """

    metadata_fnames: List[str] = [".metadata"]

    def __init__(
        self,
        dirpath: str,
        *,
        save_every_n_train_steps: Optional[int] = None,
        save_every_n_epochs: Optional[int] = None,
        save_every_n_eval_steps: Optional[int] = None,
        save_every_n_eval_epochs: Optional[int] = None,
        save_every_n_predict_steps: Optional[int] = None,
        keep_last_n_checkpoints: Optional[int] = None,
        best_checkpoint_config: Optional[BestCheckpointConfig] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        async_checkpoint: bool = False,
        knob_options: Optional[KnobOptions] = None,
    ) -> None:
        super().__init__(
            dirpath=dirpath,
            save_every_n_train_steps=save_every_n_train_steps,
            save_every_n_epochs=save_every_n_epochs,
            save_every_n_eval_steps=save_every_n_eval_steps,
            save_every_n_eval_epochs=save_every_n_eval_epochs,
            save_every_n_predict_steps=save_every_n_predict_steps,
            keep_last_n_checkpoints=keep_last_n_checkpoints,
            best_checkpoint_config=best_checkpoint_config,
            process_group=process_group,
        )
        self._async_checkpoint = async_checkpoint

        self._knob_options: KnobOptions = knob_options or KnobOptions()
        self._prev_snapshot: Optional[Future] = None

    def _checkpoint_impl(
        self,
        state: State,
        unit: AppStateMixin,
        *,
        checkpoint_id: str,
        hook: str,
        planner: Optional[SavePlanner] = None,
        storage_writer: Optional[StorageWriter] = None,
        stager: Optional[AsyncStager] = None,
    ) -> bool:
        if hook not in [
            "on_train_step_end",
            "on_train_epoch_end",
            "on_train_end",
            "on_eval_epoch_end",
            "on_eval_step_end",
            "on_predict_step_end",
            "on_predict_end",
        ]:
            raise RuntimeError(f"Unexpected hook encountered '{hook}'")

        # intra epoch when checkpointing during "_step_end" hook OR
        # when checkpointing during "on_eval_epoch_end" hook and the entry point is fit
        # since it is still intra epoch with respect to the train epoch
        intra_epoch = "step_end" in hook or (
            "on_eval_epoch_end" == hook and state.entry_point == EntryPoint.FIT
        )
        curr_snapshot_wait = hook in ("on_train_end", "on_predict_end")

        if planner is None:
            planner = DefaultSavePlanner()

        if storage_writer is None:
            storage_writer = Writer(checkpoint_id, **self.default_writer_options)

        app_state = _prepare_app_state_for_checkpoint(state, unit, intra_epoch)
        # TODO: evaluate whether we need to implement the equivalent of torchsnapshot.RNGState()
        if self._async_checkpoint:
            with get_timing_context(state, f"{self.__class__.__name__}.async_save"):
                # Redundant check for safety
                self._wait(log_warning=True)
                prev_snapshot = dcp.async_save(
                    state_dict={"app_state": MultiStateful(app_state)},
                    checkpoint_id=checkpoint_id,
                    process_group=self._process_group,
                    storage_writer=storage_writer,
                    planner=planner,
                    async_stager=stager,
                )
                self._prev_snapshot = cast(Future, prev_snapshot)
                if curr_snapshot_wait:
                    self._wait(log_warning=False)
        else:
            with get_timing_context(state, f"{self.__class__.__name__}.save"):
                dcp.save(
                    state_dict={"app_state": MultiStateful(app_state)},
                    checkpoint_id=checkpoint_id,
                    process_group=self._process_group,
                    storage_writer=storage_writer,
                    planner=planner,
                )

        return True

    def _wait(self, log_warning: bool = True) -> None:
        """
        If the previous async checkpoint is still running, wait for it to finish before continuing. Otherwise,
        distributed collectives that use the checkpointing process group will result in a stuck job. This also
        computes and logs the time spent waiting on the previous checkpoint to finish, and a toggable warning
        for the user to modify checkpointing frequency.

        If the previous checkpoint has already finished, this is a no-op.

        Args:
            log_warning: Toggle for logging a warning to the user to modify checkpointing frequency. Sometimes
                this is not up to the user (e.g. on_exception, on_train_end).
        """
        if self._prev_snapshot is None:
            return

        if self._prev_snapshot.done():
            none_throws(self._prev_snapshot).result()
            return

        if log_warning:
            rank_zero_warn(
                (
                    "Waiting on previous checkpoint to finish... Consider modifying checkpointing "
                    f"frequency if this is an issue. Current value (current {self._save_every_n_train_steps})"
                ),
                logger=logger,
            )

        t0 = time.monotonic()
        none_throws(self._prev_snapshot).result()

        rank_zero_warn(
            f"Waiting on previous checkpoint for {time.monotonic()-t0:.3f} seconds",
            logger=logger,
        )

    def on_exception(
        self,
        state: State,
        unit: Union[TTrainUnit, TEvalUnit, TPredictUnit],
        exc: BaseException,
    ) -> None:
        rank_zero_info("Ensuring previous async checkpoint finished before exiting.")
        self._wait(log_warning=False)

    @staticmethod
    def restore(
        path: str,
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        eval_dataloader: Optional[Iterable[TEvalData]] = None,
        predict_dataloader: Optional[Iterable[TPredictData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
        knob_options: Optional[KnobOptions] = None,
        planner: Optional[LoadPlanner] = None,
        storage_reader: Optional[StorageReader] = None,
        **kwargs: Any,
    ) -> None:
        """Utility method to restore dcp checkpoint from a path."""

        checkpoint_id = path

        DistributedCheckpointSaver.restore_with_id(
            checkpoint_id,
            unit,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            predict_dataloader=predict_dataloader,
            process_group=process_group,
            restore_options=restore_options,
            knob_options=knob_options,
            planner=planner,
            storage_reader=storage_reader,
        )

    @staticmethod
    def restore_with_id(
        checkpoint_id: Union[int, str],
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        eval_dataloader: Optional[Iterable[TEvalData]] = None,
        predict_dataloader: Optional[Iterable[TPredictData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
        knob_options: Optional[KnobOptions] = None,
        planner: Optional[LoadPlanner] = None,
        storage_reader: Optional[StorageReader] = None,
        **kwargs: Any,
    ) -> None:
        """Utility method to restore dcp checkpoint from a checkpoint_id.

        There are additional flags offered should the user want to skip loading the train and eval progress.
        By default, the train and eval progress are restored, if applicable.

        Args:
            checkpoint_id: Checkpoint id. It can be the path of the snapshot to restore.
            unit: An instance of :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit` containing states to restore.
            train_dataloader: An optional train dataloader to restore. Can only be used when restoring from a train or fit checkpoint.
            eval_dataloader: An optional eval dataloader to restore. Can only be used when restoring from an eval or fit checkpoint.
            predict_dataloader: An optional predict dataloader to restore. Can only be used when restoring from a predict checkpoint.
            process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
                            If not Gloo, a Gloo process group is created.
                            Note: If torch.distributed is available and a process group is initialized, dcp assumes the intention is to save/load checkpoints in distributed fashion.
            restore_options: Controls what to  filter when restoring the state.
            knob_options: Additional keyword options for StorageWriter and StorageReader
            planner: Instance of LoadPlanner. If this is not specified, the default planner will be used. (Default: ``None``)
            storage_reader: Instance of StorageReader used to perform reads. If this is not specified, it will automatically infer
                            the reader based on the checkpoint_id. If checkpoint_id is also None, an exception will be raised. (Default: ``None``)
        """
        checkpoint_id = str(checkpoint_id)
        restore_options = restore_options or RestoreOptions()

        try:
            checkpoint_path = CheckpointPath.from_str(checkpoint_id)
        except ValueError:
            checkpoint_path = None

        # For evaluation and prediction checkpoints, don't load model parameters
        if (
            checkpoint_path
            and checkpoint_path._is_phase_aware()
            and Phase.TRAIN not in checkpoint_path.step
        ):
            logger.info(
                f"Will not restore model parameters, optimizers, or lr schedulers from {checkpoint_id}, "
                + "since it is a prediction or evaluation checkpoint."
            )
            restore_options.restore_modules = False
            restore_options.restore_optimizers = False
            restore_options.restore_lr_schedulers = False

        app_state = _prepare_app_state_for_restore(unit, restore_options)

        # If no storage_reader is provided, default to path based reader
        if storage_reader is None:
            storage_reader = Reader(checkpoint_id)

        # If no planner is provided, use the default planner
        if planner is None:
            allow_partial_load = not restore_options.strict
            planner = DefaultLoadPlanner(allow_partial_load=allow_partial_load)

        # Add the dataloader to the app state using the correct key if it exists
        DistributedCheckpointSaver._maybe_add_dataloader_to_app_state(
            app_state,
            checkpoint_path,
            storage_reader,
            train_dataloader,
            eval_dataloader,
            predict_dataloader,
        )

        if restore_options.init_optim_states:
            # if optimizers states are initialized lazy
            for obj in app_state.values():
                # sometimes optimizers are actually held in a wrapper which handles calling
                # state_dict and load_state_dict, sa is the case for
                # `torchtnt.utils.prepare_module.FSDPOptimizerWrapper`, this handles that case.
                optimizer = getattr(obj, "optimizer", obj)
                if isinstance(optimizer, torch.optim.Optimizer):
                    _init_optim_state(optimizer)

        with get_or_create_gloo_pg(candidate_pg=process_group) as pg:
            dcp.load(
                {"app_state": MultiStateful(app_state)},
                checkpoint_id=checkpoint_id,
                storage_reader=storage_reader,
                planner=planner,
                process_group=pg,
            )

        rank_zero_info(
            f"Restored the checkpoint with checkpoint_id: {checkpoint_id}",
            logger=logger,
        )

    @staticmethod
    def _maybe_add_dataloader_to_app_state(
        app_state: Dict[str, Stateful],
        checkpoint_path: Optional[CheckpointPath],
        storage_reader: StorageReader,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        eval_dataloader: Optional[Iterable[TEvalData]] = None,
        predict_dataloader: Optional[Iterable[TPredictData]] = None,
    ) -> Dict[str, Stateful]:
        """
        Appends the dataloader state to the application state.
        """
        dataloaders = {
            Phase.TRAIN: train_dataloader,
            Phase.EVALUATE: eval_dataloader,
            Phase.PREDICT: predict_dataloader,
        }

        # Consider only the dataloaders passed by the user, and only if they are Stateful
        ignore_phases = set()
        for phase, dl in dataloaders.items():
            if dl is None:
                ignore_phases.add(phase)

            elif not isinstance(dl, Stateful):
                logger.warning(
                    f"dataloader for {phase} phase was passed to `restore` but it does not implement the Stateful protocol to load states"
                )
                ignore_phases.add(phase)

        candidate_dataloaders = {
            phase: dl for phase, dl in dataloaders.items() if phase not in ignore_phases
        }

        if not candidate_dataloaders:
            return app_state

        # check if any of the candidate phases has a dataloader saved in the checkpoint. If so, include
        # it in the app state. More than one dataloader may be present if using fit.
        metadata = storage_reader.read_metadata()
        for key in metadata.state_dict_metadata.keys():
            for phase in candidate_dataloaders.keys():
                if (dl_key := _PHASE_DL_STATE_KEY_MAPPING[phase]) in key:
                    if dl_key not in app_state:
                        app_state[dl_key] = cast(
                            Stateful, none_throws(candidate_dataloaders[phase])
                        )
                        logger.info(f"Restoring dataloader with key: {dl_key}")

        if dl_missing_in_ckpt := [
            phase
            for phase in candidate_dataloaders.keys()
            if _PHASE_DL_STATE_KEY_MAPPING[phase] not in app_state
        ]:
            logger.warning(
                f"dataloader ({','.join(str(k) for k in dl_missing_in_ckpt)}) was passed to `restore` "
                "but no dataloader exists in checkpoint metadata."
            )

        return app_state

    def _generate_checkpoint_and_upkeep(
        self, state: State, unit: Union[TTrainUnit, TEvalUnit, TPredictUnit], hook: str
    ) -> bool:
        # if we are still checkpointing, this might cause a collective hang, since several
        # operations in the base class use the process group. So wait here instead.
        self._wait()

        # Note that every async checkpoint will be completed at this point.
        return super()._generate_checkpoint_and_upkeep(state, unit, hook)

    @property
    def default_writer_options(self) -> Dict[str, Any]:
        # defaults are picked to to match TSS defaults
        # TODO: expose these options in KnobOptions
        dcp_options = {
            "thread_count": self._knob_options.max_per_rank_io_concurrency or 16,
            "sync_files": False,
            "overwrite": True,
        }
        if dcp_options["thread_count"] > 1:
            dcp_options["single_file_per_rank"] = False

        return dcp_options
