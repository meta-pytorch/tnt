# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Callable, Dict, List, MutableMapping, Optional, Tuple

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import EntryPoint, State
from torchtnt.framework.unit import AppStateMixin, TEvalUnit, TPredictUnit, TTrainUnit
from torchtnt.utils.module_summary import (
    get_module_summary,
    get_summary_table,
    ModuleSummary as ModuleSummaryObj,
    prune_module_summary,
)
from torchtnt.utils.rank_zero_log import rank_zero_info


def _log_module_summary_tables(module_summaries: List[ModuleSummaryObj]) -> None:
    for ms in module_summaries:
        rank_zero_info("\n" + get_summary_table(ms))


class ModuleSummary(Callback):
    """
    A callback which generates and logs a summary of the modules.

    Args:
        max_depth: The maximum depth of module summaries to keep.
        process_fn: Function to print the module summaries. Default is to log all module summary tables.
        module_inputs: A mapping from module name to (args, kwargs) for that module. Useful when wanting FLOPS, activation sizes, etc.

    Raises:
        RuntimeError:
            If torcheval is not installed.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        process_fn: Callable[
            [List[ModuleSummaryObj]], None
        ] = _log_module_summary_tables,
        module_inputs: Optional[
            MutableMapping[str, Tuple[Tuple[Any, ...], Dict[str, Any]]]
        ] = None,
    ) -> None:
        self._max_depth = max_depth
        self._process_fn = process_fn
        self._module_inputs = module_inputs

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        self._get_and_process_summaries(unit)

    def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
        if state.entry_point != EntryPoint.EVALUATE:
            return
        self._get_and_process_summaries(unit)

    def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
        self._get_and_process_summaries(unit)

    def _retrieve_module_summaries(self, unit: AppStateMixin) -> List[ModuleSummaryObj]:
        module_summaries = []
        for module_name, module in unit.tracked_modules().items():
            args, kwargs = (), {}
            if self._module_inputs and module_name in self._module_inputs:
                args, kwargs = self._module_inputs[module_name]
            module_summary = get_module_summary(
                module, module_args=args, module_kwargs=kwargs
            )
            module_summary._module_name = module_name
            if self._max_depth:
                prune_module_summary(module_summary, max_depth=self._max_depth)
            module_summaries.append(module_summary)
        return module_summaries

    def _get_and_process_summaries(self, unit: AppStateMixin) -> None:
        module_summaries = self._retrieve_module_summaries(unit)
        self._process_fn(module_summaries)
