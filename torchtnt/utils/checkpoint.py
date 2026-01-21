# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import bisect
import logging
import math
import os
import re
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering
from operator import xor
from typing import Any, Dict, List, Literal, Optional, Pattern, Tuple, Union

import fsspec
import torch
import torch.distributed as dist
from fsspec.core import url_to_fs
from pyre_extensions import none_throws
from torch import nn
from torch.distributed.tensor import distribute_tensor
from torch.nn.modules.module import _IncompatibleKeys
from torchtnt.utils.distributed import PGWrapper, rank_zero_read_and_broadcast

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class MetricData:
    """
    Representation of a metric instance. Should provide both a metric name and its value.

    Note: The metric name is sanitized by replacing '/' with '_' to prevent potential issues
        when using the name as a path or identifier.
    """

    name: str
    value: float

    def __init__(self, name: str, value: float) -> None:
        self.name = MetricData.sanitize_metric_name(name)
        self.value = value

    @classmethod
    def sanitize_metric_name(cls, name: str) -> str:
        """
        Sanitizes a metric name by replacing '/' with '_'.
        This is done to prevent potential issues when using the name as a path or identifier.
        """
        return name.replace("/", "_")


@dataclass
class BestCheckpointConfig:
    """
    Config for saving the best checkpoints.

    Args:
        monitored_metric: Metric to monitor for saving best checkpoints. Must be an numerical or tensor attribute on the unit.
        mode: One of `min` or `max`. The save file is overwritten based the max or min of the monitored metric.
    """

    monitored_metric: str
    mode: Literal["min", "max"] = "min"


class Phase(Enum):
    NONE = 0  # Only used for backwards compatibility
    TRAIN = 1
    EVALUATE = 2
    PREDICT = 3

    def __str__(self) -> str:
        return self.name.lower()


@total_ordering
class CheckpointPath:
    """
    Representation of a checkpoint path. Handles parsing and serialization of the specific path format.

    A CheckpointPath can be metric aware/naive, and/or phase aware/naive. This means:
    - A metric aware checkpoint stores the value of a particular metric name. It is possible to compare optimality of two
        checkpoints that are tracking the same metric by using the `is_more_optimal_than` method.
    - A phase aware checkpoint stores the step number for a particular phase. This provides a better UX when doing manual
        exploration of the generated checkpoints. The phase will also be considered to determine recency. However, note that
        it is possible to compare phase naive and phase aware checkpoints by recency, with the latter always being newer. For
        two phase aware checkpoints, first the evaluation steps (if any) are compared, then the train steps. It is always assumed
        that evaluations happens after training for a particular epoch.

    Examples of compliant checkpoint paths:
    - phase-naive and metric-naive- <dirpath>/epoch_<epoch>_step_<step>
    - phase-naive and metric-aware- <dirpath>/epoch_<epoch>_step_<step>_<metric_name>=<metric_value>
    - phase-aware (train only) and metric-naive- <dirpath>/epoch_<epoch>_train_step_<train_step>_<metric_name>=<metric_value>
    - phase-aware and metric-aware- <dirpath>/epoch_<epoch>_train_step_<train_step>_eval_step_<eval_step>_<metric_name>=<metric_value>
    """

    _FLOAT_REGEX: str = r"-?\d+\.?\d*(?:e[\+\-]\d+)?"

    PHASE_NAIVE_REGEX: Pattern = re.compile(
        rf"^(.+)epoch_(\d+)_step_(\d+)(?:_(\w+)=({_FLOAT_REGEX}))?\/?$"
    )

    PHASE_AWARE_REGEX: Pattern = re.compile(
        rf"^(.+)epoch_(\d+)(?:_train_step_(\d+))?(?:_eval_step_(\d+))?(?:_predict_step_(\d+))?(?:_(\w+)=({_FLOAT_REGEX}))?\/?$"
    )

    def __init__(
        self,
        dirpath: str,
        epoch: int,
        step: Union[Dict[Phase, int], int],
        metric_data: Optional[MetricData] = None,
    ) -> None:
        """
        Args:
            dirpath: The base directory path that checkpoints are saved in.
            epoch: The epoch number of this checkpoint.
            step: The step number of this checkpoint. This may be an integer for a phase-naive checkpoint. To
                create a phase-aware path, a dict can be used to map each phase to its step number.
            metric_data: Optional data about the metric being tracked. Should contain both metric name and value.
                Note that NaN and inf values are not allowed, since the signal an error in a computation and
                optimality comparisons will not work correctly.
        """
        self.dirpath: str = dirpath.rstrip("/")
        self.epoch = epoch
        self.metric_data = metric_data
        self.step: Dict[Phase, int] = (
            step if isinstance(step, dict) else {Phase.NONE: step}
        )

        if metric_data and math.isnan(metric_data.value):
            raise ValueError(
                f"Value of monitored metric '{metric_data.name}' can't be NaN in CheckpointPath."
            )

        if metric_data and math.isinf(metric_data.value):
            raise ValueError(
                f"Value of monitored metric '{metric_data.name}' can't be inf in CheckpointPath."
            )

    @classmethod
    def from_str(cls, checkpoint_path: str) -> "CheckpointPath":
        """
        Given a directory path, try to parse it and extract the checkpoint data.
        The expected format is: <dirpath>/epoch_<epoch>_step_<step>_<metric_name>=<metric_value>,
        where the metric name and value are optional.

        Args:
            checkpoint_path: The path to the checkpoint directory.

        Returns:
            A CheckpointPath instance if the path is valid, otherwise None.

        Raises:
            ValueError: If the path is malformed and can't be parsed.
        """
        ckpt_path = cls.__new__(cls)
        ckpt_path._populate_from_str(checkpoint_path)
        return ckpt_path

    def _populate_from_str(self, checkpoint_path: str) -> None:
        """
        Reusable method to parse a checkpoint path string, extract the components, and populate
        a checkpoint path instance.

        Args:
            checkpoint_path: The checkpoint path string.

        Raises:
            ValueError: If the path is malformed (either non-parsable, or contains wrong data types)
        """
        is_phase_aware = any(
            phase in checkpoint_path
            for phase in ["train_step", "eval_step", "predict_step"]
        )
        regex = self.PHASE_AWARE_REGEX if is_phase_aware else self.PHASE_NAIVE_REGEX
        path_match = regex.match(checkpoint_path)
        if not path_match:
            raise ValueError(
                f"Attempted to parse malformed checkpoint path: {checkpoint_path}."
            )

        try:
            step_mapping: Dict[Phase, int] = {}
            if is_phase_aware:
                (
                    dirpath,
                    epoch,
                    train_steps,
                    eval_steps,
                    predict_steps,
                    metric_name,
                    metric_value,
                ) = path_match.groups()

                if train_steps is not None:
                    step_mapping[Phase.TRAIN] = int(train_steps)
                if eval_steps is not None:
                    step_mapping[Phase.EVALUATE] = int(eval_steps)
                if predict_steps is not None:
                    step_mapping[Phase.PREDICT] = int(predict_steps)

            else:
                dirpath, epoch, naive_steps, metric_name, metric_value = (
                    path_match.groups()
                )
                step_mapping[Phase.NONE] = int(naive_steps)

            metric_data: Optional[MetricData] = None
            if metric_name:
                metric_value_f = float(metric_value)
                metric_data = MetricData(name=metric_name, value=metric_value_f)

            self.dirpath = dirpath.rstrip("/")
            self.epoch = int(epoch)
            self.step = step_mapping
            self.metric_data = metric_data

        except ValueError:
            # Should never happen since path matches regex
            raise ValueError(
                f"Invalid data types found in checkpoint path: {checkpoint_path}."
            )

    @property
    def path(self) -> str:
        """
        Returns:
            The full path to the checkpoint directory.
        """
        name = f"epoch_{self.epoch}"

        if not self._is_phase_aware():
            name += f"_step_{self.step[Phase.NONE]}"
        else:
            if Phase.TRAIN in self.step:
                name += f"_train_step_{self.step[Phase.TRAIN]}"
            if Phase.EVALUATE in self.step:
                name += f"_eval_step_{self.step[Phase.EVALUATE]}"
            if Phase.PREDICT in self.step:
                name += f"_predict_step_{self.step[Phase.PREDICT]}"

        if self.metric_data:
            name += f"_{self.metric_data.name}={self.metric_data.value}"

        return os.path.join(self.dirpath, name)

    def _is_phase_aware(self) -> bool:
        return Phase.NONE not in self.step

    def newer_than(self, other: "CheckpointPath") -> bool:
        """
        Given another CheckpointPath instance, determine if this checkpoint is strictly newer than the other.
        Note that recency is determined in terms of the epoch, phase, and number of steps. It is NOT related to
        the timestamp the checkpoint was saved.

        Returns:
            True if this checkpoint is newer than the other, otherwise False.
        """
        # First check always the epoch
        if self.epoch != other.epoch:
            return self.epoch > other.epoch

        # If comparing a phase-aware vs phase-naive checkpoint, determine recency by the total number of steps.
        # In case both are the same, the phase aware checkpoint is considered newer.
        if xor(self._is_phase_aware(), other._is_phase_aware()):
            if sum(self.step.values()) != sum(other.step.values()):
                return sum(self.step.values()) > sum(other.step.values())
            return self._is_phase_aware()

        # For two phase-naive checkpoints, we only need to look at the step
        if not self._is_phase_aware():
            return self.step[Phase.NONE] > other.step[Phase.NONE]

        # If one checkpoint has eval steps and the other doesn't, the one with eval steps is always newer
        if xor(Phase.EVALUATE in self.step, Phase.EVALUATE in other.step):
            return Phase.EVALUATE in self.step

        # Otherwise, compare first by eval and then train steps
        return self._get_phase_steps() > other._get_phase_steps()

    def _get_phase_steps(self) -> Tuple[int, ...]:
        """Tuple with the phase steps ordered by phase priority in comparison (predict, eval, train)."""
        return (
            self.step.get(Phase.PREDICT, 0),
            self.step.get(Phase.EVALUATE, 0),
            self.step.get(Phase.TRAIN, 0),
        )

    def more_optimal_than(
        self, other: "CheckpointPath", mode: Literal["min", "max"]
    ) -> bool:
        """
        Given another CheckpointPath instance, determine if this checkpoint is strictly more optimal than the other.
        Optimality is determined by comparing the metric value of the two checkpoints. The mode indicates if the
        metric value should be minimized or maximized. This only works for metric-aware checkpoints.

        Args:
            other: The other checkpoint path to compare against.
            mode: The mode to use for comparison.

        Returns:
            True if this checkpoint is more optimal than the other, otherwise False.

        Note: This expects that both checkpoints are metric-aware, and that they are tracking the same metric.
              If one of these is not metric aware, the one which is metric aware will be chosen. If both are
              not metric aware, the newer one will be chosen.
        """

        if self.metric_data and not other.metric_data:
            return True
        elif other.metric_data and not self.metric_data:
            return False
        elif not self.metric_data and not other.metric_data:
            return self.newer_than(other)

        assert (
            none_throws(self.metric_data).name == none_throws(other.metric_data).name
        ), f"Attempted to compare optimality of checkpoints tracking different metrics: {self} and {other}"

        if mode == "min":
            return (
                none_throws(self.metric_data).value
                < none_throws(other.metric_data).value
            )

        return (
            none_throws(self.metric_data).value > none_throws(other.metric_data).value
        )

    def __str__(self) -> str:
        return self.path

    def __repr__(self) -> str:
        return f"CheckpointPath(dirpath={self.dirpath}, epoch={self.epoch}, step={self.step}, metric_data={self.metric_data})"

    def __eq__(self, other: "CheckpointPath") -> bool:
        return (
            self.dirpath == other.dirpath
            and self.epoch == other.epoch
            and self.step == other.step
            and self.metric_data == other.metric_data
        )

    def __gt__(self, other: "CheckpointPath") -> bool:
        return self.newer_than(other)

    def __getstate__(self) -> str:
        # Lightweight pickling to avoid broadcast errors
        return self.path

    def __setstate__(self, state: str) -> None:
        # Match regex directly to avoid creating a new instance with `from_str`
        self._populate_from_str(state)


class CheckpointManager:
    """
    Manage a group of CheckpointPaths that belong to the same base directory. This involves maintaining
    ordering checkpoints by recency or metric value if applicable. Then, this is used to determine if a
    checkpoint should be saved, and what name will be used.

    The checkpoints work in the following format: <dirpath>/epoch_<epoch>_step_<step>
    If a metric is being tracked, it's added to the name: <dirpath>/epoch_<epoch>_step_<step>_<metric_name>=<metric_value>

    The methods in this class are meant to be used in the following order:
    1. Create instance - this will load the existing checkpoints (if any)
    2. `prune_surplus_checkpoints` - this will remove the non-optimal checkpoints to enforce the `keep_last_n_checkpoints`
    3. For every checkpointing iteration:
        a. `generate_checkpoint_path`: Gives the CheckpointPath that would be saved next
        b. `should_save_checkpoint`: Determines if checkpoint should be saved according to the `keep_last_n_checkpoints` and `best_checkpoint_config`
        c. -- The external checkpointing API should be called if above returns True. CheckpointManager does NOT actually generate checkpoints --
        d. `append_checkpoint`: If the checkpoint was successfully saved, this should be called to update the internal state

    In general, every file system read/write operation performed by this class is executed only in rank 0, while state is synced across ranks.
    """

    def __init__(
        self,
        dirpath: str,
        best_checkpoint_config: Optional[BestCheckpointConfig] = None,
        keep_last_n_checkpoints: Optional[int] = None,
        metadata_fnames: Optional[List[str]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        file_system: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """
        Initialize a checkpoint manager. If a `keep_last_n_checkpoints` value is provided, this will read the
        existing checkpoints in the dirpath (from rank 0 only) to account for them in the max number of checkpoints
        to keep. Note that no checkpoints are deleted.

        Args:
            dirpath: The base directory path that checkpoints are saved in. This is synced from rank 0 to every other rank upon initialization.
            best_checkpoint_config: Optional configuration for the best checkpoint.
            keep_last_n_checkpoints: Optional number of checkpoints to keep.
            metadata_fname: Optional names of the metadata files. These are used to verify checkpoint integrity. If more than one is provided, a
                checkpoint is considered if at least one of them exists.
            process_group: Optional process group to use for distributed training. gloo process groups are known
                to perform better.
        """
        self.dirpath: str = self._sync_dirpath_to_all_ranks(
            dirpath=dirpath, process_group=process_group
        )

        self._best_checkpoint_config = best_checkpoint_config
        self._keep_last_n_checkpoints = keep_last_n_checkpoints
        self._pg_wrapper = PGWrapper(process_group)

        if file_system is None:
            file_system, _ = url_to_fs(self.dirpath)

        self._file_system: fsspec.AbstractFileSystem = file_system

        if metadata_fnames is None:
            self._metadata_fnames: List[str] = []
        else:
            self._metadata_fnames = metadata_fnames

        self._ckpt_paths: List[CheckpointPath] = []
        if not self._keep_last_n_checkpoints:
            return

        # If there is a max limit of checkpoints to store, keep track of existing ones
        metric_name = (
            best_checkpoint_config.monitored_metric if best_checkpoint_config else None
        )
        self._ckpt_paths = get_checkpoint_dirpaths(
            dirpath=dirpath,
            metadata_fname=self._metadata_fnames,
            metric_name=metric_name,
            process_group=process_group,
        )
        if best_checkpoint_config:
            self._ckpt_paths.sort(
                key=lambda x: x.metric_data.value,
                # sort descending if min, placing worst metric at top of list
                reverse=(best_checkpoint_config.mode == "min"),
            )
        else:
            self._ckpt_paths.sort()  # Checkpoint paths are well-ordered by recency

    def prune_surplus_checkpoints(self) -> None:
        """
        Prune checkpoints that exceed the maximum number of checkpoints to keep. This should be
        called when training starts, so that the `keep_last_n_checkpoints` config is honored.
        Files are only deleted in rank 0.

        Note:
            This is not called on initialization, in case users want to inpsect previous
            checkpoints. But it should be called before starting training if there is a
            `keep_last_n_checkpoints` config.

        Args:
            state: The training state.
        """
        keep_last_n_checkpoints = self._keep_last_n_checkpoints
        if keep_last_n_checkpoints and len(self._ckpt_paths) > keep_last_n_checkpoints:
            logger.warning(
                (
                    f"{len(self._ckpt_paths)} checkpoints found in {self.dirpath}. ",
                    f"Deleting {len(self._ckpt_paths) - keep_last_n_checkpoints} oldest ",
                    "checkpoints to enforce ``keep_last_n_checkpoints`` argument.",
                )
            )
            for _ in range(len(self._ckpt_paths) - keep_last_n_checkpoints):
                self.remove_checkpoint()

    @rank_zero_read_and_broadcast
    def generate_checkpoint_path(
        self,
        epoch: int,
        step: Union[int, Dict[Phase, int]],
        metric_data: Optional[MetricData] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> CheckpointPath:
        """
        Given the current epoch, step, and possibly a metric_data value, determine the path
        where it should be stored. This does not necessarily mean that the checkpoint should
        be created. Instead, `should_save_checkpoint` has to be called to determine that.

        Args:
            unit: The training unit.
            state: The training state.

        Returns:
            The path to the checkpoint to save.

        Raises: AssertionError if there is a mismatch in tracked metric, for example:
            - `best_checkpoint_config` is not set but `metric_data` was provided
            - `best_checkpoint_config` is set and `metric_data` is passed. But they are not tracking the same metric
        """

        if metric_data:
            assert (
                self._best_checkpoint_config
            ), "Attempted to get a checkpoint with metric but best checkpoint config is not set"

            assert (
                MetricData.sanitize_metric_name(
                    self._best_checkpoint_config.monitored_metric
                )
                == metric_data.name
            ), (
                f"Attempted to get a checkpoint with metric '{metric_data.name}', "
                f"but best checkpoint config is for '{MetricData.sanitize_metric_name(none_throws(self._best_checkpoint_config).monitored_metric)}'"
            )

        checkpoint_path = CheckpointPath(
            self.dirpath, epoch, step, metric_data=metric_data
        )

        return checkpoint_path

    def should_save_checkpoint(self, checkpoint: CheckpointPath) -> bool:
        """
        Given a unit and state, determine if a checkpoint should be saved when considering the `keep_last_n_checkpoints`
        and `best_checkpoint_config` configs.

        Args:
            checkpoint: The CheckpointPath to be potentially saved, provided by `generate_checkpoint_path`.

        Returns:
            True if the checkpoint should be saved, otherwise False.
        """

        keep_last_n_checkpoints = self._keep_last_n_checkpoints
        if not keep_last_n_checkpoints:
            # always save candidate checkpoint if no limit of checkpoints is specified
            return True

        if len(self._ckpt_paths) < keep_last_n_checkpoints:
            # limit of checkpoints has not been reached
            return True

        best_checkpoint_config = self._best_checkpoint_config
        if not best_checkpoint_config:
            # we always save the latest checkpoint
            return True

        # otherwise, we need to determine if we should overwrite the worst checkpoint
        return checkpoint.more_optimal_than(
            self._ckpt_paths[0], mode=best_checkpoint_config.mode
        )

    def get_last_checkpoint(self) -> Optional[CheckpointPath]:
        """
        Returns last checkpoint to be added
        """
        if len(self._ckpt_paths) == 0:
            return None
        return self._ckpt_paths[-1]

    def append_checkpoint(self, ckpt: CheckpointPath) -> None:
        """
        This will update the internal state to keep track of the checkpoint. This function should only be called
        when a checkpoint whose path was returned from `generate_checkpoint_path` was successfully created.
        If a previous checkpoint should be removed to honor `keep_last_n_checkpoint`, it will be deleted on rank 0.

        Args:
            ckpt: The checkpoint to save.
            state: The training state.
        """
        # Remove oldest/worst checkpoint if needed
        max_ckpts = self._keep_last_n_checkpoints
        if max_ckpts and len(self._ckpt_paths) >= max_ckpts:
            self.remove_checkpoint()

        # If we are monitoring a metric, but the checkpoint has no metric data, we don't track it
        if self._best_checkpoint_config and ckpt.metric_data:
            keys = [none_throws(c.metric_data).value for c in self._ckpt_paths]
            if self._best_checkpoint_config.mode == "min":
                keys.reverse()

            # Use bisect.bisect() to find the insertion point
            idx = bisect.bisect(keys, none_throws(ckpt.metric_data).value)
            if none_throws(self._best_checkpoint_config).mode == "min":
                idx = len(self._ckpt_paths) - idx
            self._ckpt_paths.insert(idx, ckpt)

        elif not self._best_checkpoint_config:
            # No metric tracked, most recents goes last
            self._ckpt_paths.append(ckpt)

    def does_checkpoint_exist(
        self,
        ckpt: CheckpointPath,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> bool:
        """
        Checking whether a checkpoint already exists by verifying whether the optional metadata file is present in the directory.
        If the checkpointer doesn't have a metadata file, this function will always return False. Check is executed in rank 0, but
        result is broadcasted to all ranks.
        """
        return does_checkpoint_exist(
            ckpt.path, self._metadata_fnames, process_group=process_group
        )

    def does_checkpoint_metadata_exist(
        self,
        checkpoint_path: str,
        metadata_fname: str,
    ) -> bool:
        """
        Checking whether a checkpoint metadata file exists in the directory.
        If the checkpointer has that metadata file, this function will returns True. Returns False otherwise.
        """
        return _metadata_exists(self._file_system, checkpoint_path, metadata_fname)

    @staticmethod
    @rank_zero_read_and_broadcast
    def _sync_dirpath_to_all_ranks(
        dirpath: str, process_group: Optional[dist.ProcessGroup] = None
    ) -> str:
        """Synchronize the dirpath across all ranks."""
        return dirpath

    def remove_checkpoint(self) -> None:
        """
        Delete the weakest checkpoint both from the internal state and from the file system (rank 0). This means:
        - If there is a `best_checkpoint_config`, then the checkpoint with the least optimal metric value
        - If there is no `best_checkpoint_config`, then the oldest checkpoint
        """
        worst_ckpt_path = self._ckpt_paths.pop(0)
        if self._pg_wrapper.get_rank() == 0:
            try:
                self._file_system.rm(worst_ckpt_path.path, recursive=True)
            except Exception as exc:
                logger.error(
                    (
                        f"Failed to remove checkpoint '{worst_ckpt_path}' for bookkeeping purposes. "
                        f"Do not use it to restore since it may be corrupted! Exception: {exc}"
                    )
                )


@rank_zero_read_and_broadcast
def does_checkpoint_exist(
    ckpt_path: str,
    metadata_fname: Union[str, List[str]],
    file_system: Optional[fsspec.AbstractFileSystem] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> bool:
    """
    Checking whether a checkpoint already exists by verifying whether the optional metadata file is present in the directory.
    Will return False if the metadata_fname is None. Check is executed in rank 0, but
    result is broadcasted to all ranks.

    Args:
        ckpt: The checkpoint to check.
        metadata_fname: File to check for existence. If a list is provided, it will check that at least one of the files is present.
        file_system: If a custom file system should be used to fetch the checkpoint directories. Otherwise, fsspec will be
            used to match the file system of the dirpath.
        process_group: Optional process group on which the ranks will communicate on. By default, the entire world is used.
    """
    if not metadata_fname:
        return False
    else:
        metadata_fnames = (
            [metadata_fname] if isinstance(metadata_fname, str) else metadata_fname
        )

    fs = file_system
    if fs is None:
        fs, _ = url_to_fs(ckpt_path)

    return any(_metadata_exists(fs, ckpt_path, fname) for fname in metadata_fnames)


@rank_zero_read_and_broadcast
def get_latest_checkpoint_path(
    dirpath: str,
    metadata_fname: Optional[Union[str, List[str]]] = None,
    file_system: Optional[fsspec.AbstractFileSystem] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Optional[str]:
    """
    Given a parent directory where checkpoints are saved, return the latest checkpoint subdirectory.

    Args:
        dirpath: parent directory where checkpoints are saved.
        metadata_fname: Checks if metadata file is present in checkpoint, disregards if it does not exist.
                    If a list is provided, it will check that at least one of the files is present.
        file_system: If a custom file system should be used to fetch the checkpoint directories. Otherwise, fsspec will be
            used to match the file system of the dirpath.
        process_group: the process group on which the ranks will communicate on. default: ``None`` (the entire world)

    Raises:
        AssertionError if the checkpoint subdirectories are not named in the format epoch_{epoch}_step_{step}.

    Note:
        When doing distributed training, only rank 0 will read the file system. The result will be broadcasted to all ranks.
        gloo process groups are recommended over nccl.
    """

    return _get_latest_checkpoint_path(dirpath, metadata_fname, file_system)


def _get_latest_checkpoint_path(
    dirpath: str,
    metadata_fname: Optional[Union[str, List[str]]] = None,
    file_system: Optional[fsspec.AbstractFileSystem] = None,
) -> Optional[str]:
    candidate_dirpaths = _retrieve_checkpoint_dirpaths(
        dirpath, metadata_fname, file_system=file_system
    )
    if not candidate_dirpaths:
        return None

    latest_checkpoint = candidate_dirpaths[0]
    for candidate in candidate_dirpaths[1:]:
        if candidate.newer_than(latest_checkpoint):
            latest_checkpoint = candidate

    return latest_checkpoint.path


@rank_zero_read_and_broadcast
def get_best_checkpoint_path(
    dirpath: str,
    metric_name: str,
    mode: Literal["min", "max"],
    metadata_fname: Optional[Union[str, List[str]]] = None,
    file_system: Optional[fsspec.AbstractFileSystem] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Optional[str]:
    """
    Given a parent directory where checkpoints are saved, return the best checkpoint subdirectory based on a metric.

    The checkpoint paths are assumed to have the following format: <dirpath>/epoch_<epoch>_step_<step>_<metric_name>=<metric_value>
    This will always be the case if the CheckpointManager class is used to produce their names.

    Args:
        dirpath: parent directory where checkpoints are saved.
        metric_name: Name of the metric to use to find the best checkpoint.
        mode: Either 'min' or 'max'. If 'min', finds and loads the lowest value metric checkpoint. If 'max', finds and loads the largest.
        metadata_fname: Checks if metadata file is present in checkpoint, disregards if it does not exist.
                    If a list is provided, it will check that at least one of the files is present.
        file_system: If a custom file system should be used to fetch the checkpoint directories. Otherwise, fsspec will be
            used to match the file system of the dirpath.
        process_group: the process group on which the ranks will communicate on. default: ``None`` (the entire world)

    Note:
        When doing distributed training, only rank 0 will read the file system. The result will be broadcasted to all ranks.
        gloo process groups are recommended over nccl.
    """

    dirpaths = _retrieve_checkpoint_dirpaths(
        dirpath, metadata_fname, metric_name, file_system=file_system
    )
    if not dirpaths:
        return None

    best_checkpoint = dirpaths[0]
    for checkpoint in dirpaths[1:]:
        if checkpoint.more_optimal_than(best_checkpoint, mode):
            best_checkpoint = checkpoint

    return best_checkpoint.path


@rank_zero_read_and_broadcast
def get_checkpoint_dirpaths(
    dirpath: str,
    metadata_fname: Optional[Union[str, List[str]]] = None,
    metric_name: Optional[str] = None,
    file_system: Optional[fsspec.AbstractFileSystem] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> List[CheckpointPath]:
    """
    Given a parent directory where checkpoints are saved, returns the checkpoint subdirectories.
    The order of the checkpoints is not guaranteed.

    The checkpoint paths are assumed to have the following format: <dirpath>/epoch_<epoch>_step_<step>
    If a metric_name is provided the format should be <dirpath>/epoch_<epoch>_step_<step>_<metric_name>=<metric_value>
    This will always be the case if the CheckpointManager class is used to produce their names.

    Args:
        dirpath: parent directory where checkpoints are saved.
        metadata_fname: Checks if metadata file is present in checkpoint, disregards if it does not exist.
                    If a list is provided, it will check that at least one of the files is present.
        metric_name: fetches all the checkpoint directories containing the metric name only.
        file_system: If a custom file system should be used to fetch the checkpoint directories. Otherwise, fsspec will be
            used to match the file system of the dirpath.
        process_group: the process group on which the ranks will communicate on. default: ``None`` (the entire world)

    Note:
        When doing distributed training, only rank 0 will read the file system. The result will be broadcasted to all ranks.
        gloo process groups are recommended over nccl.
    """

    return _retrieve_checkpoint_dirpaths(
        dirpath, metadata_fname, metric_name, file_system=file_system
    )


def _retrieve_checkpoint_dirpaths(
    dirpath: str,
    metadata_fname: Optional[Union[str, List[str]]],
    metric_name: Optional[str] = None,
    file_system: Optional[fsspec.AbstractFileSystem] = None,
) -> List[CheckpointPath]:
    """
    Given a parent directory where checkpoints are saved, return the unsorted checkpoint subdirectories

    Args:
        dirpath: parent directory where checkpoints are saved.
        metadata_fname: Checks if metadata file is present in checkpoint, disregards if it does not exist.
            If a list is provided, it will check that at least one of the files is present.
        metric_name: Name of the metric that must exist in checkpoint name.
        file_system: If a custom file system should be used to fetch the checkpoint directories. Otherwise, fsspec will be
            used to match the file system of the dirpath.
    """
    fs = file_system
    if fs is None:
        fs, _ = url_to_fs(dirpath)

    if not fs.exists(dirpath):
        logger.warning(f"Input dirpath doesn't exist: {dirpath}")
        return []

    contents = fs.ls(dirpath, detail=True)
    contents = [item["name"] for item in contents if item["type"] == "directory"]
    if len(contents) == 0:
        logger.warning(f"Input dirpath doesn't contain any subdirectories: {dirpath}")
        return []

    # Parse the valid checkpoint directories
    candidate_checkpoints: List[CheckpointPath] = []
    for candidate_dirpath in contents:
        try:
            ckpt = CheckpointPath.from_str(candidate_dirpath)
        except ValueError:
            continue

        # If a metric was provided, keep only the checkpoints tracking it
        if metric_name and not (
            ckpt.metric_data
            and ckpt.metric_data.name == MetricData.sanitize_metric_name(metric_name)
        ):
            continue

        candidate_checkpoints.append(ckpt)

    if not metadata_fname:
        # return early as we don't need to filter out any paths
        return candidate_checkpoints

    # Iterate through all files and directories in the specified directory
    # and check if metadata is present or not
    metadata_fnames = (
        [metadata_fname] if isinstance(metadata_fname, str) else metadata_fname
    )
    valid_ckpt_dirpaths: List[CheckpointPath] = []
    for candidate in candidate_checkpoints:
        if any(
            _metadata_exists(fs, candidate.path, fname) for fname in metadata_fnames
        ):
            valid_ckpt_dirpaths.append(candidate)
            continue

        logger.warning(
            f"Snapshot metadata ({metadata_fnames}) missing from {candidate}! Skipping this path"
        )

    return valid_ckpt_dirpaths


def _metadata_exists(
    fs: fsspec.AbstractFileSystem, dirpath: str, metadata_fname: str
) -> bool:
    return fs.exists(os.path.join(dirpath, metadata_fname))


def load_from_full_model_state_dict(
    model: torch.nn.Module,
    full_sd: Dict[str, Any],
    device: torch.device,
    strict: bool = False,
    cpu_offload: bool = False,
    release_sd: bool = True,
) -> _IncompatibleKeys:
    """
    Converting full state dict into a sharded state dict
    and loading it into FSDP model
    Args:
        model (Module): Model to generate fully qualified names for cpu_state_dict
        full_sd (dict[str, Any]): a full state dict to load into the model (mmap=True for efficient loading)
        device (torch.device): device used to move full state dict tensors
        strict (bool): flag to check if to load the model in strict mode
        cpu_offload (bool): flag to check if offload to CPU is enabled
        release_sd (bool): whether to release memory of full_sd to save ram usage
    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys
    """
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    for param_name, full_tensor in sorted(full_sd.items()):
        sharded_meta_param = meta_sharded_sd.get(param_name)
        assert sharded_meta_param is not None, f"{param_name} not found in model"
        full_tensor = full_tensor.to(sharded_meta_param.dtype).to(device)
        if not hasattr(sharded_meta_param, "device_mesh"):
            # In cases where parts of the model aren't sharded, some parameters will be plain tensors
            sharded_tensor = full_tensor
        else:
            sharded_tensor = distribute_tensor(
                full_tensor,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
            )
        if cpu_offload:
            sharded_tensor = sharded_tensor.cpu()
        sharded_sd[param_name] = nn.Parameter(sharded_tensor)
        if release_sd:
            full_sd[param_name] = None
    # choose `assign=True` since we cannot call `copy_` on meta tensor
    return model.load_state_dict(sharded_sd, strict=strict, assign=True)
