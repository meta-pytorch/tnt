# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

try:
    from torch._inductor.async_compile import shutdown_compile_workers
except ImportError:

    def shutdown_compile_workers() -> None:
        logging.warning(
            "shutdown_compile_workers is not available in your version of PyTorch. \
            Please use nightly version to enable this feature."
        )


from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TTrainUnit

logger: logging.Logger = logging.getLogger(__name__)


class TorchCompile(Callback):
    """
    A callback for using torch.compile.

    Args:
        step_shutdown_compile_workers: step after which compiler workers
        will be shut down.
    """

    def __init__(self, step_shutdown_compile_workers: int) -> None:
        self._step_shutdown_compile_workers = step_shutdown_compile_workers

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        total_num_steps_completed = unit.train_progress.num_steps_completed
        if total_num_steps_completed == self._step_shutdown_compile_workers:
            logger.info(
                f"Shutdown compile workers after step {total_num_steps_completed}"
            )
            shutdown_compile_workers()
