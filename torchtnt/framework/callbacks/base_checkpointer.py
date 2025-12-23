# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import logging
import math
from datetime import timedelta
from typing import Any, cast, Dict, Iterable, List, Literal, Optional, Union

import fsspec

import fsspec

import torch.distributed as dist
from pyre_extensions import none_throws
from torchtnt.framework.callback import Callback
from torchtnt.framework.callbacks._checkpoint_utils import (
    _get_epoch,
    _get_step_phase_mapping,
)
from torchtnt.framework.callbacks.checkpointer_types import RestoreOptions
from torchtnt.framework.state import EntryPoint, State
from torchtnt.framework.unit import (
    AppStateMixin,
    TEvalUnit,
    TPredictUnit,
    TTrainData,
    TTrainUnit,
)
from torchtnt.utils.checkpoint import (
    BestCheckpointConfig,
    CheckpointManager,
    get_best_checkpoint_path,
    get_latest_checkpoint_path,
    MetricData,
    Phase,
)
from torchtnt.utils.distributed import get_world_size, PGWrapper
from torchtnt.utils.event_handlers import log_interval
from torchtnt.utils.rank_zero_log import rank_zero_info, rank_zero_warn

logger: logging.Logger = logging.getLogger(__name__)


class BaseCheckpointer(Callback, metaclass=abc.ABCMeta):
    """
    Abstract base class for file-based state_dict checkpointing. This class can be used as the base of a checkpointing callback, and handles
    checkpointing frequency logic, checkpoint naming, checkpoint purging / upkeep, and process group synchronization. There are only two methods
    that need to be implemented by subclasses:

    1) ``_checkpoint_impl`` which implements the checkpoint saving logic, given the relevant checkpoint items and path.
    2) ``restore`` which implements restoring the checkpoint given the relevant checkpoint path.

    The subclass may override the ``metadata_fname`` attribute to specify the filename of the metadata file that will be written within the checkpoint directory.
    This will be used by this base class to ensure the integrity of the checkpoint. This is a list because some checkpointers may allow more than one valid
    ``metadata_fnames``, depending on storage or optimization configurations.

    If running in a distributed environment, the default process group should be initialized prior to instantiating this Callback. This is done automatically if
    using `AutoUnit`, which should be instantiated first.

    Args:
        dirpath: Parent directory to save checkpoints to.
        save_every_n_train_steps: Frequency of steps with which to save checkpoints during the train epoch. If None, no intra-epoch checkpoints are generated.
        save_every_n_epochs: Frequency of epochs with which to save checkpoints during training. If None, no end-of-epoch checkpoints are generated.
        save_every_n_eval_epochs: Frequency of evaluation epochs with which to save checkpoints during training. Use this if wanting to save checkpoints after every eval epoch during fit.
        save_every_n_eval_steps: Frequency of evaluation steps with which to save checkpoints during training. Use this if wanting to save checkpoints during evaluate.
        save_every_n_predict_steps: Frequency of prediction steps with which to save checkpoints during training. Use this if wanting to save checkpoints during using predict entrypoint.
        keep_last_n_checkpoints: Number of most recent checkpoints to keep. If None, all checkpoints are kept. If an excess of existing checkpoints are present, the oldest ones will be deleted
            to clean the difference. If best checkpoint config is enabled, this param will manage the top n checkpoints instead. Only supported for train or fit entrypoints.
        best_checkpoint_config: Configuration for saving the best checkpoint based on a monitored metric. The metric is read off the attribute of the unit prior to checkpoint. This param is ignored if not in train or fit entrypoints.
        process_group: The process group on which the ranks will communicate on. If the process group is not gloo-based, a new gloo-based process group will be created.

    Note:
        If torch.distributed is available and default process group is initialized, the constructor will call a collective operation for rank 0 to broadcast the dirpath to all other ranks

    Note:
        This class assumes checkpoint items are saved in the directory provided in ``_checkpoint_impl`` and will be in the form of ``<dirpath>/<epoch>-<step>/``. Checkpoint contents
        should be stored within this directory, as deleting and retrieving latest checkpoint relies on reading the <epoch>-<step> directory name within <dirpath>

    Note:
        If best_checkpoint_config is enabled, the attribute must be on the unit upon checkpoint time, and must be castable to "float". This value must be maintained by the unit, and updated
        appropriately. For example, if logging validation accuracy, the unit must be responsible for maintaining the value and resetting it when the epoch ends. If the metric value is None, the
        checkpoint will be saved, without the metric value in the checkpoint name
    """

    # No metadata file is checked by default. This can be overridden by subclasses.
    metadata_fnames: List[str] = []

    def __init__(
        self,
        dirpath: str,
        *,
        save_every_n_train_steps: Optional[int] = None,
        save_every_n_epochs: Optional[int] = None,
        save_every_n_eval_epochs: Optional[int] = None,
        save_every_n_eval_steps: Optional[int] = None,
        save_every_n_predict_steps: Optional[int] = None,
        keep_last_n_checkpoints: Optional[int] = None,
        best_checkpoint_config: Optional[BestCheckpointConfig] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        if get_world_size() > 1 and not dist.is_initialized():
            raise RuntimeError(
                "Running in a distributed environment without default process group initialized. "
                "Call `torch.distributed.init_process_group` before initializing this callback. "
                "Using `AutoUnit` will do this automatically."
            )

        if save_every_n_train_steps is not None and save_every_n_train_steps <= 0:
            raise ValueError(
                f"Invalid value passed for save_every_n_train_steps. Expected to receive either None or positive number, but received {save_every_n_train_steps}"
            )
        if save_every_n_epochs is not None and save_every_n_epochs <= 0:
            raise ValueError(
                f"Invalid value passed for save_every_n_epochs. Expected to receive either None or positive number, but received {save_every_n_epochs}"
            )
        if save_every_n_eval_steps is not None and save_every_n_eval_steps <= 0:
            raise ValueError(
                f"Invalid value passed for save_every_n_eval_steps. Expected to receive either None or positive number, but received {save_every_n_eval_steps}"
            )
        if save_every_n_eval_epochs is not None and save_every_n_eval_epochs <= 0:
            raise ValueError(
                f"Invalid value passed for save_every_n_eval_epochs. Expected to receive either None or positive number, but received {save_every_n_eval_epochs}"
            )
        if save_every_n_predict_steps is not None and save_every_n_predict_steps <= 0:
            raise ValueError(
                f"Invalid value passed for save_every_n_predict_steps. Expected to receive either None or positive number, but received {save_every_n_predict_steps}"
            )
        if keep_last_n_checkpoints is not None and keep_last_n_checkpoints <= 0:
            raise ValueError(
                f"Invalid value passed for keep_last_n_checkpoints. Expected to receive either None or positive number, but received {keep_last_n_checkpoints}"
            )

        if best_checkpoint_config and best_checkpoint_config.mode not in {"min", "max"}:
            raise ValueError(
                f"Invalid value passed for best_checkpoint_config.mode. Expected to receive 'min' or 'max', but received {best_checkpoint_config.mode}"
            )

        self._save_every_n_train_steps = save_every_n_train_steps
        self._save_every_n_epochs = save_every_n_epochs
        self._save_every_n_eval_epochs = save_every_n_eval_epochs
        self._save_every_n_eval_steps = save_every_n_eval_steps
        self._save_every_n_predict_steps = save_every_n_predict_steps
        self._keep_last_n_checkpoints = keep_last_n_checkpoints
        self._best_checkpoint_config = best_checkpoint_config

        self._process_group: Optional[dist.ProcessGroup] = None
        self._setup_gloo_pg(process_group)
        self._pg_wrapper = PGWrapper(process_group)

        self._checkpoint_manager = CheckpointManager(
            dirpath,
            best_checkpoint_config,
            keep_last_n_checkpoints,
            metadata_fnames=self.metadata_fnames,
            process_group=self._process_group,
        )

    def _setup_gloo_pg(self, process_group: Optional[dist.ProcessGroup]) -> None:
        """
        Setups gloo process group to be used for any collectives called during
        checkpointing. If global process group is already gloo, no action is required.
        Gloo is used over nccl for better compatibility.
        """
        if not dist.is_initialized():
            # there can be no process group
            return

        if process_group is None:
            # use global process group
            process_group = dist.group.WORLD

        # we create a new gloo process group if different backend is being used
        if dist.get_backend(process_group) != dist.Backend.GLOO:
            rank_zero_info("Creating new gloo process group for checkpointing.")
            self._process_group = dist.new_group(
                timeout=timedelta(seconds=3600), backend=dist.Backend.GLOO
            )
        else:
            self._process_group = process_group

    @property
    def dirpath(self) -> str:
        """Returns parent directory to save to."""
        return self._checkpoint_manager.dirpath

    def _generate_checkpoint_and_upkeep(
        self, state: State, unit: Union[TTrainUnit, TEvalUnit, TPredictUnit], hook: str
    ) -> bool:
        """
        Implementation for saving checkpoint while taking care of checkpoint
        name generation and cleanup of oldest checkpoints.

        Args:
            state: Current state of the trainer.
            unit: Current training unit.
            hook: Hook at which checkpoint is being saved.

        Returns:
            True if checkpoint was successfully saved. False otherwise.
        """
        log_interval_metadata: Dict[str, str] = {
            "category": "checkpointing",
            "active_phase": str(state.active_phase),
            "hook": hook,
            "epoch": str(_get_epoch(state, unit)),
            "step": str(
                _get_step_phase_mapping(state, unit).get(
                    state.active_phase.into_phase(), 0
                )
            ),
        }

        with log_interval(
            "_generate_checkpoint_and_upkeep", metadata=log_interval_metadata
        ):
            # 1) generate checkpoint name
            epoch = _get_epoch(state, unit)
            step_mapping = _get_step_phase_mapping(state, unit)

            # 1.1) append metric data only if best_checkpoint_config is defined
            metric_data: Optional[MetricData] = None
            if self._best_checkpoint_config and (
                metric_value := self._get_tracked_metric_value(cast(TTrainUnit, unit))
            ):
                metric_data = MetricData(
                    name=none_throws(self._best_checkpoint_config).monitored_metric,
                    value=metric_value,
                )

            checkpoint_path = self._checkpoint_manager.generate_checkpoint_path(
                epoch,
                step_mapping,
                metric_data,
                process_group=self._process_group,
            )

            # 2) Determine if we should save checkpoint. This is a no-op for eval and predict entrypoints
            # since neither best_checkpoint_config nor keep_last_n_checkpoints are supported.
            if not self._checkpoint_manager.should_save_checkpoint(checkpoint_path):
                return False

            if hook == "on_train_end":
                # 2.1) Make sure that last checkpoint does not already exist
                if self._checkpoint_manager.does_checkpoint_exist(
                    checkpoint_path, self._process_group
                ):
                    rank_zero_warn(
                        "Final checkpoint already exists, skipping.", logger=logger
                    )
                    return False

                # 2.2) If doing fit without eval checkpointing, only consider training progress when
                # checking if last checkpoint exists.
                if (
                    state.entry_point == EntryPoint.FIT
                    and self._save_every_n_eval_epochs is None
                    and self._checkpoint_manager._ckpt_paths
                    and self._checkpoint_manager._ckpt_paths[-1].step[Phase.TRAIN]
                    == cast(TTrainUnit, unit).train_progress.num_steps_completed
                ):
                    rank_zero_info(
                        "Omitting final checkpoint since train progress is unchanged, and eval checkpointing is not configured.",
                        logger=logger,
                    )
                    return False

            # 3) try to save checkpoint
            if not self._checkpoint_impl(
                state, unit, checkpoint_id=checkpoint_path.path, hook=hook
            ):
                return False

            # 4) track checkpoint and clean up surplus if needed
            self._checkpoint_manager.append_checkpoint(checkpoint_path)

            # 5) invoke on_checkpoint_save callback on the unit since checkpoint was saved successfully
            unit.on_checkpoint_save(state, checkpoint_id=checkpoint_path.path)

            return True

    def _get_tracked_metric_value(self, unit: TTrainUnit) -> Optional[float]:
        """
        If the checkpointer has a tracked metric, look the value in the unit using reflection, and cast to float.

        Args:
            unit: The training unit to look for the tracked metric in.

        Returns:
            The value of the tracked metric, or None if there is no best_checkpoint config defined.

        Raises:
            RuntimeError: If the unit does not have the attribute specified in the best_checkpoint config,
                or if the value cannot be cast to a float.
        """
        if not self._best_checkpoint_config:
            return None

        monitored_metric_name = self._best_checkpoint_config.monitored_metric
        if not hasattr(unit, monitored_metric_name):
            logger.error(
                f"Unit does not have attribute {monitored_metric_name}, unable to retrieve metric to checkpoint. "
                "Will not be included in checkpoint path, nor tracked for optimality."
            )
            return None

        metric_value_f = None
        if (metric_value := getattr(unit, monitored_metric_name)) is not None:
            try:
                metric_value_f = float(metric_value)
            except ValueError as exc:
                logger.error(
                    f"Unable to convert monitored metric {monitored_metric_name} to a float: {exc}. Please ensure the value "
                    "can be converted to float and is not a multi-element tensor value. Will not be included in checkpoint path, "
                    "nor tracked for optimality."
                )
                return None

        if metric_value_f and math.isnan(metric_value_f):
            logger.error(
                f"Monitored metric '{monitored_metric_name}' is NaN. Will not be included in checkpoint path, nor tracked for optimality."
            )
            return None

        if metric_value_f and math.isinf(metric_value_f):
            logger.error(
                f"Monitored metric '{monitored_metric_name}' is inf. Will not be included in checkpoint path, nor tracked for optimality."
            )
            return None

        return metric_value_f

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        # clean up the difference if surplus of checkpoints exist
        self._checkpoint_manager.prune_surplus_checkpoints()

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        num_steps_completed = unit.train_progress.num_steps_completed
        if (
            not self._save_every_n_train_steps
            or num_steps_completed % self._save_every_n_train_steps != 0
        ):
            return

        self._generate_checkpoint_and_upkeep(state, unit, hook="on_train_step_end")

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        epoch = unit.train_progress.num_epochs_completed
        if not self._save_every_n_epochs or epoch % self._save_every_n_epochs != 0:
            return

        self._generate_checkpoint_and_upkeep(state, unit, hook="on_train_epoch_end")

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        self._generate_checkpoint_and_upkeep(state, unit, hook="on_train_end")

    def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
        if state.entry_point == EntryPoint.EVALUATE:
            self._disable_ckpt_optimality_tracking()

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        num_steps_completed = unit.eval_progress.num_steps_completed
        if (
            not self._save_every_n_eval_steps
            or num_steps_completed % self._save_every_n_eval_steps != 0
        ):
            return

        self._generate_checkpoint_and_upkeep(state, unit, hook="on_eval_step_end")

    def on_eval_epoch_end(self, state: State, unit: TEvalUnit) -> None:
        epoch = unit.eval_progress.num_epochs_completed
        if (
            not self._save_every_n_eval_epochs
            or epoch % self._save_every_n_eval_epochs != 0
        ):
            return

        self._generate_checkpoint_and_upkeep(state, unit, hook="on_eval_epoch_end")

    def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
        self._disable_ckpt_optimality_tracking()

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        num_steps_completed = unit.predict_progress.num_steps_completed
        if (
            not self._save_every_n_predict_steps
            or num_steps_completed % self._save_every_n_predict_steps != 0
        ):
            return

        self._generate_checkpoint_and_upkeep(state, unit, hook="on_predict_step_end")

    def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
        self._generate_checkpoint_and_upkeep(state, unit, hook="on_predict_end")

    def _disable_ckpt_optimality_tracking(self) -> None:
        """
        Disables checkpoint optimality tracking. This means that best_checkpoint and keep_last_n_checkpoints
        will not be used. This is useful for eval and predict entrypoints, since checkpoints do not include
        model parameters.
        """
        if self._best_checkpoint_config:
            logger.warning(
                "Disabling best_checkpoint_config, since it is not supported for eval or predict entrypoints."
            )
            self._best_checkpoint_config = None
            self._checkpoint_manager._best_checkpoint_config = None

        if self._keep_last_n_checkpoints:
            logger.warning(
                "Disabling keep_last_n_checkpoints, since is not supported for eval or predict entrypoints."
            )
            self._keep_last_n_checkpoints = None
            self._checkpoint_manager._keep_last_n_checkpoints = None

    @abc.abstractmethod
    def _checkpoint_impl(
        self,
        state: State,
        unit: AppStateMixin,
        *,
        checkpoint_id: str,
        hook: str,
    ) -> bool:
        """
        Implementation of saving checkpoint.

        Args:
            state: current application state
            unit: current unit
            checkpoint_id: Checkpoint id to save a checkpoint. It can be a path
            hook: name of callback hook that triggered this function call

        Returns:
            Whether a new checkpoint was created.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def restore(
        path: str,
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
        **kwargs: Any,
    ) -> None:
        """Method to restore checkpoint state from a path.

        There are additional flags offered should the user want to skip loading the train and eval progress.
        By default, the train and eval progress are restored, if applicable.

        Args:
            path: Path of the checkpoint to restore.
            unit: An instance of :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit` containing states to restore.
            train_dataloader: An optional train dataloader to restore. Can only be used when restoring from a train or fit checkpoint.
            process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
            restore_options: Controls what to filter when restoring the state.
        """
        ...

    @classmethod
    def restore_from_latest(
        cls,
        dirpath: str,
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
        file_system: Optional[fsspec.AbstractFileSystem] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Given a parent directory where checkpoints are saved, restore the checkpoint state from the latest checkpoint in the directory.

        There are additional flags offered should the user want to skip loading the train and eval progress.
        By default, the train and eval progress are restored, if applicable.

        Args:
            dirpath: Parent directory from which to get the latest checkpoint.
            unit: An instance of :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit` containing states to restore.
            train_dataloader: An optional train dataloader to restore.
            process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
            restore_options: Controls what to  filter when restoring the state.
            file_system: If a custom file system should be used to fetch the checkpoint directories. Otherwise, fsspec will be
                used to match the file system of the dirpath.

        Returns:
            True if the latest checkpoint directory was found and successfully restored, otherwise False.
        """
        path = get_latest_checkpoint_path(
            dirpath,
            metadata_fname=cls.metadata_fnames,
            process_group=process_group,
            file_system=file_system,
        )
        if path is None:
            logger.info(
                f"Attempted to restore from the following path but no checkpoint was found: {dirpath=}, {cls.metadata_fnames}"
            )
            return False
        logger.info(f"Restoring from path: {path}")
        cls.restore(
            path,
            unit,
            train_dataloader=train_dataloader,
            process_group=process_group,
            restore_options=restore_options,
            **kwargs,
        )
        return True

    @classmethod
    def restore_from_best(
        cls,
        dirpath: str,
        unit: AppStateMixin,
        metric_name: str,
        mode: Literal["min", "max"],
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
        file_system: Optional[fsspec.AbstractFileSystem] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Given a parent directory where checkpoints are saved, restore the checkpoint state from the best checkpoint in the directory.

        There are additional flags offered should the user want to skip loading the train and eval progress.
        By default, the train and eval progress are restored, if applicable.

        Args:
            dirpath: Parent directory from which to get the latest checkpoint.
            unit: An instance of :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit` containing states to restore.
            metric_name: Name of the metric to use to find the best checkpoint.
            mode: Either 'min' or 'max'. If 'min', finds and loads the lowest value metric checkpoint. If 'max', finds and loads the largest.
            train_dataloader: An optional train dataloader to restore.
            process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
            file_system: If a custom file system should be used to fetch the checkpoint directories. Otherwise, fsspec will be
                used to match the file system of the dirpath.
            restore_options: Controls what to  filter when restoring the state.

        Returns:
            True if the best checkpoint directory was found and successfully restored, otherwise False.
        """
        best_checkpoint_path = get_best_checkpoint_path(
            dirpath,
            metric_name=metric_name,
            mode=mode,
            metadata_fname=cls.metadata_fnames,
            file_system=file_system,
            process_group=process_group,
        )

        if best_checkpoint_path is None:
            rank_zero_warn(
                f"No checkpoints with metric name {metric_name} were found in {dirpath}. Not loading any checkpoint.",
                logger=logger,
            )
            return False

        rank_zero_info(f"Loading checkpoint from {best_checkpoint_path}")

        cls.restore(
            best_checkpoint_path,
            unit,
            train_dataloader=train_dataloader,
            process_group=process_group,
            restore_options=restore_options,
            **kwargs,
        )

        return True

    @staticmethod
    def restore_with_id(
        checkpoint_id: Union[int, str],
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
        **kwargs: Any,
    ) -> None:
        """Method to restore checkpoint state from a checkpoint id.

        There are additional flags offered should the user want to skip loading the train and eval progress.
        By default, the train and eval progress are restored, if applicable.

        This method relies on the user to provide a checkpoint id. This offers flexibility to the users
        overriding the BaseCheckpointer if they want to use a different way to represent a checkpoint.
        Default implementation of BaseCheckpointer uses the checkpoint path as id.

        Args:
            checkpoint_id: Checkpoint ID. It can be the path of the checkpoint as well to restore.
            unit: An instance of :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit` containing states to restore.
            train_dataloader: An optional train dataloader to restore.
            process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
            restore_options: Controls what to filter when restoring the state.
        """
        BaseCheckpointer.restore(
            str(checkpoint_id),
            unit,
            train_dataloader=train_dataloader,
            process_group=process_group,
            restore_options=restore_options,
            **kwargs,
        )
