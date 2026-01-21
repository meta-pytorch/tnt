# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from .anomaly_evaluation import IsNaNEvaluator, ThresholdEvaluator
from .checkpoint import (
    BestCheckpointConfig,
    CheckpointManager,
    CheckpointPath,
    get_best_checkpoint_path,
    get_checkpoint_dirpaths,
    get_latest_checkpoint_path,
    MetricData,
)
from .device import (
    copy_data_to_device,
    CPUStats,
    get_device_from_env,
    get_nvidia_smi_gpu_stats,
    get_psutil_cpu_stats,
    GPUStats,
    record_data_in_stream,
    set_float32_precision,
)
from .distributed import (
    all_gather_tensors,
    barrier,
    get_global_rank,
    get_local_rank,
    get_process_group_backend_from_device,
    get_world_size,
    PGWrapper,
    spawn_multi_process,
    sync_bool,
)
from .early_stop_checker import EarlyStopChecker
from .env import init_from_env, seed
from .flops import FlopTensorDispatchMode
from .fsspec import get_filesystem
from .lr_scheduler import TLRScheduler
from .memory import get_tensor_size_bytes_map, measure_rss_deltas, RSSProfiler
from .memory_snapshot_profiler import MemorySnapshotParams, MemorySnapshotProfiler
from .misc import days_to_secs, transfer_batch_norm_stats, transfer_weights
from .module_summary import (
    get_module_summary,
    get_summary_table,
    ModuleSummary,
    prune_module_summary,
)
from .nan import check_for_nan_or_inf, register_nan_hooks_on_whole_graph
from .oom import (
    attach_oom_observer,
    is_out_of_cpu_memory,
    is_out_of_cuda_memory,
    is_out_of_memory_error,
    log_memory_snapshot,
)
from .optimizer import extract_lr_from_optimizer, init_optim_state
from .precision import convert_precision_str_to_dtype
from .prepare_module import (
    DDPStrategy,
    FSDPStrategy,
    NOOPStrategy,
    prepare_ddp,
    prepare_fsdp,
)
from .profiler import IProfiler
from .progress import Progress
from .rank_zero_log import (
    rank_zero_critical,
    rank_zero_debug,
    rank_zero_error,
    rank_zero_info,
    rank_zero_print,
    rank_zero_warn,
)
from .stateful import Stateful
from .swa import AveragedModel
from .test_utils import get_pet_launch_config
from .timer import FullSyncPeriodicTimer, get_timer_summary, log_elapsed_time, Timer
from .tqdm import close_progress_bar, create_progress_bar, update_progress_bar
from .version import (
    get_python_version,
    get_torch_version,
    is_torch_version_geq,
    is_windows,
)

__all__ = [
    "check_for_nan_or_inf",
    "register_nan_hooks_on_whole_graph",
    "IsNaNEvaluator",
    "ThresholdEvaluator",
    "CheckpointPath",
    "MetricData",
    "get_best_checkpoint_path",
    "get_checkpoint_dirpaths",
    "get_latest_checkpoint_path",
    "BestCheckpointConfig",
    "CheckpointManager",
    "copy_data_to_device",
    "CPUStats",
    "get_device_from_env",
    "get_nvidia_smi_gpu_stats",
    "get_psutil_cpu_stats",
    "GPUStats",
    "set_float32_precision",
    "record_data_in_stream",
    "all_gather_tensors",
    "barrier",
    "get_global_rank",
    "get_local_rank",
    "get_process_group_backend_from_device",
    "get_world_size",
    "PGWrapper",
    "sync_bool",
    "EarlyStopChecker",
    "init_from_env",
    "seed",
    "FlopTensorDispatchMode",
    "get_filesystem",
    "get_tensor_size_bytes_map",
    "measure_rss_deltas",
    "RSSProfiler",
    "MemorySnapshotParams",
    "MemorySnapshotProfiler",
    "days_to_secs",
    "attach_oom_observer",
    "is_out_of_cpu_memory",
    "is_out_of_cuda_memory",
    "is_out_of_memory_error",
    "log_memory_snapshot",
    "extract_lr_from_optimizer",
    "init_optim_state",
    "convert_precision_str_to_dtype",
    "DDPStrategy",
    "FSDPStrategy",
    "NOOPStrategy",
    "prepare_ddp",
    "prepare_fsdp",
    "IProfiler",
    "Progress",
    "rank_zero_critical",
    "rank_zero_debug",
    "rank_zero_error",
    "rank_zero_info",
    "rank_zero_print",
    "rank_zero_warn",
    "Stateful",
    "AveragedModel",
    "FullSyncPeriodicTimer",
    "get_timer_summary",
    "log_elapsed_time",
    "transfer_batch_norm_stats",
    "transfer_weights",
    "get_module_summary",
    "get_summary_table",
    "ModuleSummary",
    "prune_module_summary",
    "Timer",
    "create_progress_bar",
    "close_progress_bar",
    "update_progress_bar",
    "TLRScheduler",
    "get_python_version",
    "get_torch_version",
    "is_torch_version_geq",
    "is_windows",
    "get_pet_launch_config",
    "spawn_multi_process",
]
