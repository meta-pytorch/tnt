#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Any, Callable, Dict, Optional, Union

import torch
from typing_extensions import final, Literal

_log: logging.Logger = logging.getLogger(__name__)


@final
class EarlyStopChecker:
    """
    Monitor a metric and signal if execution should stop early.

    Args:
        mode: one of `min`, `max`. In `min` mode, signal to stop early will be given when
            the metric has stopped decreasing. In `max` mode, the signal is given when the
            metric has stopped increasing.
        patience: Number of checks without improvement after which early stop will be signaled.
        min_delta: Must be >= 0. Minimum absolute or relative change in the metric to qualify as
            an improvement. In `rel` mode, improvement_threshold = best_val * ( 1 + min_delta ) in 'max'
            mode or best_val * ( 1 - min_delta ) in `min` mode. In `abs` mode, improvement_threshold =
            best_val +  min_delta in `max` mode or best_val - threshold in `min` mode.
        check_finite: When set to `True`, signals early stop when metric becomes NaN or infinite.
        threshold_mode: one of `abs` or `rel`, threshold delta between checks for determining whether to stop.
        stopping_threshold: Signals early stop once the metric improves beyond this threshold.
        divergence_threshold: Signals early stop once the metric becomes worse than this threshold.

     Raises:
        ValueError:
            If `mode` is not `min` or `max`.
        ValueError:
            If `min_delta` < 0.
        ValueError:
            If `threshold_mode` is not `abs` or `rel`.
    """

    _valid_modes = ("min", "max")
    _valid_threshold_mode = ("abs", "rel")

    def __init__(
        self,
        mode: Literal["min", "max"],
        patience: int,
        min_delta: float = 0.0,
        check_finite: bool = True,
        threshold_mode: Literal["abs", "rel"] = "abs",
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
    ) -> None:
        if mode not in self._valid_modes:
            raise ValueError(f"`mode` can be {self._valid_modes}. Got `{mode}`")

        if min_delta < 0:
            raise ValueError(
                f"`min_delta` must be greater than or equal to 0. Got {min_delta}`"
            )

        if threshold_mode not in self._valid_threshold_mode:
            raise ValueError(
                f"`threshold_mode` can be {self._valid_threshold_mode},"
                f"Got `{threshold_mode}`"
            )

        self._mode: Literal["min", "max"] = mode
        self._patience: int = patience
        self._min_delta: torch.Tensor = torch.tensor([min_delta])
        self._check_finite: bool = check_finite
        self._threshold_mode: Literal["abs", "rel"] = threshold_mode
        self._stopping_threshold: Optional[torch.Tensor] = (
            None if stopping_threshold is None else torch.tensor([stopping_threshold])
        )
        self._divergence_threshold: Optional[torch.Tensor] = (
            None
            if divergence_threshold is None
            else torch.tensor([divergence_threshold])
        )
        self._min_delta *= 1 if self._mode_func == torch.gt else -1

        # Initialize self._patience_count and self._best_value
        self.reset()

    @property
    def mode(self) -> Literal["min", "max"]:
        return self._mode

    @property
    def patience(self) -> int:
        return self._patience

    @property
    def min_delta(self) -> torch.Tensor:
        return self._min_delta

    @property
    def check_finite(self) -> bool:
        return self._check_finite

    @property
    def threshold_mode(self) -> Literal["abs", "rel"]:
        return self._threshold_mode

    @property
    def stopping_threshold(self) -> Optional[torch.Tensor]:
        return self._stopping_threshold

    @property
    def divergence_threshold(self) -> Optional[torch.Tensor]:
        return self._divergence_threshold

    def reset(self) -> None:
        """Reset back to the default state."""
        self._patience_count: int = 0
        torch_inf: torch.Tensor = torch.tensor([float("inf")])
        self._best_value: torch.Tensor = (
            torch_inf if self._mode_func == torch.lt else -torch_inf
        )

    def state_dict(self) -> Dict[str, Any]:
        """
        Generates a `state_dict` to save the current state.
        This `state_dict` can be reloaded using `load_state_dict()`.
        """
        return {
            "patience_count": self._patience_count,
            "best_value": self._best_value,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Loads the current state from a `state_dict`.
        This `state_dict` can be generated from `state_dict()`.
        """
        self._patience_count = state_dict["patience_count"]
        self._best_value = state_dict["best_value"]

    def check(self, val: Union[torch.Tensor, float, int]) -> bool:
        """
        Check the current value of a metric and determine whether to stop or not.

        Args:
            val: The metric that will be monitored to signal an early stop.
                This should be either a single element tensor or a float.

        Returns:
            A boolean indicating whether execution should stop early or not.

        Raises:
            ValueError:
                If `val` is a tensor that does not contain 1 element.
        """
        if isinstance(val, float) or isinstance(val, int):
            val = torch.tensor([float(val)])
        if val.numel() != 1:
            raise ValueError(
                f"Expected tensor with only 1 element, but input has number of elements = {val.numel()}"
            )

        # Ensure tensor dim=1 and dtype=float32 for consistent shape in state_dict save/load paths
        val = val.float().reshape(1)

        should_stop = False
        message = "No stopping conditions were satisfied."

        stopping_threshold = self.stopping_threshold
        if stopping_threshold:
            stopping_threshold = stopping_threshold.to(val.device)
        divergence_threshold = self.divergence_threshold
        if divergence_threshold:
            divergence_threshold = divergence_threshold.to(val.device)
        improvement_threshold = self.min_delta
        if self._threshold_mode == "rel":
            base_val = (
                self._best_value.to(val.device)
                if torch.isfinite(self._best_value)
                else 0.0
            )
            improvement_threshold = self.min_delta.to(val.device) * base_val

        # Check finite
        if self.check_finite and not torch.isfinite(val):
            _log.warning(
                f"Metric is not finite: {val}."
                f" Previous best value was {self._best_value}."
            )
            return True

        # Check if reached stopping threshold
        if stopping_threshold is not None and self._mode_func(val, stopping_threshold):
            _log.warning(
                "Stopping threshold reached:"
                f" {val} {self._mode_char} {stopping_threshold}."
            )
            return True

        # Check if exceeding divergence threshold
        if divergence_threshold is not None and self._mode_func(
            -val, -divergence_threshold
        ):
            _log.warning(
                "Divergence threshold reached:"
                f" {val} {self._mode_char} {divergence_threshold}."
            )
            return True

        # Check if improvement is happening
        if self._mode_func(
            val - improvement_threshold.to(val.device), self._best_value.to(val.device)
        ):
            # Still improving
            should_stop = False
            message = self._improvement_message(val)
            # Snapshot the value independently of a caller's reusable metric tensor.
            self._best_value = val.detach().clone()
            self._patience_count = 0
            _log.debug(message)

        else:
            # Not improving
            self._patience_count += 1
            # Check if patience has run out
            if self._patience_count >= self.patience:
                # Patience has run out
                should_stop = True
                message = (
                    f"Metric did not improve in the last {self._patience_count} checks."
                    f" Best score: {self._best_value}."
                )
            else:
                # Patience still remaining
                should_stop = False
                message = (
                    f"Metric did not improve in the last {self._patience_count} checks."
                    f" {self.patience - self._patience_count} checks of patience remaining."
                )

            _log.warning(message)

        return should_stop

    @property
    def _mode_func(
        self,
    ) -> Callable[[torch.Tensor, Union[torch.Tensor, float]], torch.Tensor]:
        """The comparison function to use based on the mode."""
        return torch.lt if self.mode == "min" else torch.gt

    @property
    def _mode_char(
        self,
    ) -> Literal["<", ">"]:
        """The comparison character to use based on the mode."""
        return "<" if self.mode == "min" else ">"

    def _improvement_message(self, val: torch.Tensor) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if torch.isfinite(self._best_value):
            improvement = (
                torch.abs(self._best_value.to(val.device) - val)
                if self.threshold_mode == "abs"
                else torch.abs(
                    (self._best_value.to(val.device) - val)
                    / (1.0 * self._best_value.to(val.device))
                )
            )
            msg = (
                f"Metric improved by {self.threshold_mode} {improvement} >="
                f" min_delta =  {torch.abs(self.min_delta)}. New best score: {val}"
            )
        else:
            msg = f"Metric improved. New best score: {val}"
        return msg
