# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import Optional


# TODO: eventually support overriding all knobs
@dataclass
class KnobOptions:
    """
    Controls the knobs for Checkpoints.

    Args:
        max_per_rank_io_concurrency: Maximum number of concurrent IO operations per rank in checkpointing.
                                     Defaults to 16.
        enable_storage_optimization: Enable storage efficiency optimizations for Distributed Checkpointing.
        use_collectives: If ``False``, skip the cross-rank gather/scatter rounds that DCP uses to
                         coordinate the global save plan. Each rank writes its local shards
                         independently. Defaults to ``True``. Setting to ``False`` removes the
                         coord-side bottleneck that becomes prohibitively expensive at large scales
                         (e.g. ~50 minutes of a ~67 minute save wall at 397B TP=8 EP=16). Trade-off:
                         the resulting checkpoint is no longer cross-world re-shardable; restart
                         with the same parallelism layout still works.
    """

    # use a more conservative number of concurrent IO operations per rank in Checkpointing
    # the default value of 16 is too bandwidth hungry for most users
    max_per_rank_io_concurrency: Optional[int] = None
    # This would enable storage efficiency optimizations (model store):
    # e.g. Compression, Batching, Quantization etc.
    enable_storage_optimization: bool = True
    # Skip DCP coord-side gather/scatter rounds.
    use_collectives: bool = True


@dataclass
class RestoreOptions:
    """
    Options when restoring a snapshot.

    Args:
        restore_modules: Whether to restore the module state dict.
        restore_train_progress: Whether to restore the training progress state.
        restore_eval_progress: Whether to restore the evaluation progress state.
        restore_predict_progress: Whether to restore the prediction progress state.
        restore_optimizers: Whether to restore the optimizer states.
        restore_lr_schedulers: Whether to restore the lr scheduler states.
        restore_metrics: Whether to restore the metric states.
        strict: Whether to strictly restore app state and the module state dict.
        init_optim_states: Whether to initialize the optimizer state. Defaults to True. Toggle off
            if running into issues with loading optimizer state. This will reset optimizer state,
            which may affect training in some cases.
    """

    restore_modules: bool = True
    restore_train_progress: bool = True
    restore_eval_progress: bool = True
    restore_predict_progress: bool = True
    restore_optimizers: bool = True
    restore_lr_schedulers: bool = True
    restore_metrics: bool = True
    strict: bool = True
    init_optim_states: bool = True
