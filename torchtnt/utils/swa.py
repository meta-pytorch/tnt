# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Callable, List, Literal, Optional

import torch
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

_AVERAGED_MODEL_AVAIL: bool = True

try:
    from torch.optim.swa_utils import (
        AveragedModel as PyTorchAveragedModel,
        get_ema_multi_avg_fn,
        get_swa_multi_avg_fn,
    )
except ImportError:
    _AVERAGED_MODEL_AVAIL = False


TSWA_avg_fn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
TSWA_multi_avg_fn = Callable[[List[torch.Tensor], List[torch.Tensor], int], None]


class AveragedModel(PyTorchAveragedModel):
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        use_buffers: bool = False,
        averaging_method: Literal["ema", "swa"] = "ema",
        ema_decay: float = 0.999,
        skip_deepcopy: bool = False,
        use_lit: bool = False,
    ) -> None:
        """
        This class is a custom version of AveragedModel that allows us to skip the
        automatic deepcopy step and streamline the use of EMA / SWA. The deepcopy
        optionality gives flexibility to use modules that are not
        compatible with deepcopy, like FSDP wrapped modules. Check out
        https://github.com/pytorch/pytorch/blob/main/torch/optim/swa_utils.py#L66
        to see what the model, device, and use_buffer arguments entail.

        Args:
            use_buffers: if ``True``, it will compute running averages for
                both the parameters and the buffers of the model. (default: ``False``)
                This will update activation statistics for Batch Normalization.
            averaging_method: Whether to use EMA or SWA.
            ema_decay: The exponential decay applied to the averaged parameters. This param
                is only needed for EMA, and is ignored otherwise (for SWA).
            skip_deepcopy: If True, will skip the deepcopy step. The user must ensure
                that the module passed in is already copied in someway
            use_lit: If True, will use Lit EMA style by adjusting weight decay based on the
                number of updates. The EMA decay will start small and will approach the
                specified ema_decay as more updates occur.
        """
        if not _AVERAGED_MODEL_AVAIL:
            raise ImportError(
                "AveragedModel is not available in this version of PyTorch. \
                Please install the latest version of PyTorch."
            )

        # setup averaging method
        if averaging_method == "ema":
            if ema_decay < 0.0 or ema_decay > 1.0:
                raise ValueError(f"Decay must be between 0 and 1, got {ema_decay}")

            multi_avg_fn = get_ema_multi_avg_fn(ema_decay)
        elif averaging_method == "swa":
            multi_avg_fn = get_swa_multi_avg_fn()

            if use_lit:
                raise ValueError("LitEMA is only supported for EMA.")
        else:
            raise ValueError(
                f"Unknown averaging method: {averaging_method}. Only ema and swa are supported."
            )

        self._ema_decay = ema_decay
        self._use_lit = use_lit
        self._num_updates = 0

        if skip_deepcopy:
            # calls parent init manually, but skips deepcopy step
            torch.nn.Module.__init__(self)  # inits grandparent class

            self.module: torch.nn.Module = model
            self.register_buffer(
                "n_averaged", torch.tensor(0, dtype=torch.long, device=device)
            )
            # pyrefly: ignore [bad-override-mutable-attribute]
            self.avg_fn: Optional[TSWA_avg_fn] = None
            # pyrefly: ignore [bad-override-mutable-attribute]
            self.multi_avg_fn: Optional[TSWA_multi_avg_fn] = multi_avg_fn
            self.use_buffers: bool = use_buffers
        else:
            # use default init implementation

            super().__init__(
                model,
                device=device,
                multi_avg_fn=multi_avg_fn,
                use_buffers=use_buffers,
            )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        output = self.module(*args, **kwargs)

        # for fsdp modules, we need to manually reshard the swa_model in case the
        # model fwd was used in evaluation loop, due to how fsdp manages the param state
        # see https://github.com/pytorch/pytorch/issues/117742
        for m in FullyShardedDataParallel.fsdp_modules(self.module):
            if m._has_params and m.sharding_strategy is not ShardingStrategy.NO_SHARD:
                # pyre-ignore: Incompatible parameter type [6]: In call `torch.distributed.fsdp._runtime_utils._reshard`, for 2nd positional argument, expected `FlatParamHandle` but got `Optional[FlatParamHandle]`.
                torch.distributed.fsdp._runtime_utils._reshard(m, m._handle, True)

        return output

    def update_parameters(self, model: torch.nn.Module) -> None:
        self._num_updates += 1
        if self._use_lit:
            decay = min(
                self._ema_decay, (1 + self._num_updates) / (10 + self._num_updates)
            )

            self.multi_avg_fn = get_ema_multi_avg_fn(decay)
        super().update_parameters(model)
