# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# ignore errors due to `Any` type

import logging
from enum import auto, Enum
from typing import Generic, Iterable, Optional, TypeVar

from pyre_extensions import none_throws
from torchtnt.utils.checkpoint import Phase
from torchtnt.utils.timer import BoundedTimer, TimerProtocol

_logger: logging.Logger = logging.getLogger(__name__)

TStepOutput = TypeVar("TStepOutput")
TData = TypeVar("TData")


def _check_loop_condition(name: str, val: Optional[int]) -> None:
    if val is not None and val < 0:
        raise ValueError(
            f"Invalid value provided for {name}. Expected a non-negative integer or None, but received {val}."
        )


class EntryPoint(Enum):
    """
    Enum for the user-facing functions offered by the TorchTNT framework.
    - :py:func:`~torchtnt.framework.fit`
    - :py:func:`~torchtnt.framework.train`
    - :py:func:`~torchtnt.framework.evaluate`
    - :py:func:`~torchtnt.framework.predict`
    """

    FIT = auto()
    TRAIN = auto()
    EVALUATE = auto()
    PREDICT = auto()


class ActivePhase(Enum):
    """Enum for the currently active phase.

    This class complements :class:`EntryPoint` by specifying the active phase for each function.
    More than one phase value can be set while a :class:`EntryPoint` is running:
        - ``EntryPoint.FIT`` - ``ActivePhase.{TRAIN,EVALUATE}``
        - ``EntryPoint.TRAIN`` - ``ActivePhase.TRAIN``
        - ``EntryPoint.EVALUATE`` - ``ActivePhase.EVALUATE``
        - ``EntryPoint.PREDICT`` - ``ActivePhase.PREDICT``

    This can be used within hooks such as :meth:`~torchtnt.framework.unit._OnExceptionMixin.on_exception`
    to determine within which of training, evaluation, or prediction the hook is being called.
    """

    TRAIN = auto()
    EVALUATE = auto()
    PREDICT = auto()

    def into_phase(self) -> Phase:
        """Converts the active phase to the corresponding phase."""
        if self == ActivePhase.TRAIN:
            return Phase.TRAIN
        elif self == ActivePhase.EVALUATE:
            return Phase.EVALUATE
        elif self == ActivePhase.PREDICT:
            return Phase.PREDICT
        else:
            raise AssertionError("Should match an ActivePhase")

    def __str__(self) -> str:
        if self == ActivePhase.TRAIN:
            return "train"
        elif self == ActivePhase.EVALUATE:
            return "eval"
        elif self == ActivePhase.PREDICT:
            return "predict"
        else:
            raise AssertionError("Should match an ActivePhase")


class PhaseState(Generic[TData, TStepOutput]):
    """State for each phase (train, eval, predict).
    Modified by the framework, read-only for the user.
    """

    def __init__(
        self,
        *,
        dataloader: Iterable[TData],
        max_epochs: Optional[int] = None,  # used only for train
        max_steps: Optional[int] = None,  # used only for train
        max_steps_per_epoch: Optional[int] = None,
        evaluate_every_n_steps: Optional[int] = None,  # used only for evaluate
        evaluate_every_n_epochs: Optional[int] = None,  # used only for evaluate
    ) -> None:
        _check_loop_condition("max_epochs", max_epochs)
        _check_loop_condition("max_steps", max_steps)
        _check_loop_condition("max_steps_per_epoch", max_steps_per_epoch)
        _check_loop_condition("evaluate_every_n_steps", evaluate_every_n_steps)
        _check_loop_condition("evaluate_every_n_epochs", evaluate_every_n_epochs)

        self._dataloader: Iterable[TData] = dataloader
        self._max_epochs = max_epochs
        self._max_steps = max_steps
        self._max_steps_per_epoch = max_steps_per_epoch
        self._evaluate_every_n_steps = evaluate_every_n_steps
        self._evaluate_every_n_epochs = evaluate_every_n_epochs

        self._step_output: Optional[TStepOutput] = None
        self._iteration_timer = BoundedTimer(
            cuda_sync=False, lower_bound=1_000, upper_bound=5_000
        )

    @property
    def dataloader(self) -> Iterable[TData]:
        """Dataloader defined by the user."""
        return self._dataloader

    @property
    def max_epochs(self) -> Optional[int]:
        """Maximum number of epochs to train, defined by the user."""
        return self._max_epochs

    @property
    def max_steps(self) -> Optional[int]:
        """Maximum number of steps to train, defined by the user."""
        return self._max_steps

    @property
    def max_steps_per_epoch(self) -> Optional[int]:
        """Maximum number of steps to run per epoch, defined by the user."""
        return self._max_steps_per_epoch

    @property
    def evaluate_every_n_steps(self) -> Optional[int]:
        """Frequency with which to evaluate in terms of training steps, when running :func:`~torchtnt.framework.fit`. Defined by the user."""
        return self._evaluate_every_n_steps

    @evaluate_every_n_steps.setter
    def evaluate_every_n_steps(self, value: Optional[int]) -> None:
        _check_loop_condition("evaluate_every_n_steps", value)
        self._evaluate_every_n_steps = value

    @property
    def evaluate_every_n_epochs(self) -> Optional[int]:
        """Frequency with which to evaluate in terms of training epochs, when running :func:`~torchtnt.framework.fit`. Defined by the user."""
        return self._evaluate_every_n_epochs

    @evaluate_every_n_epochs.setter
    def evaluate_every_n_epochs(self, value: Optional[int]) -> None:
        _check_loop_condition("evaluate_every_n_epochs", value)
        self._evaluate_every_n_epochs = value

    @property
    def step_output(self) -> Optional[TStepOutput]:
        """Output of the last step."""
        return self._step_output

    @property
    def iteration_timer(self) -> TimerProtocol:
        """An always-on :class:`~torchtnt.utils.TimerProtocol` object which contains CPU timings (without synchronisation) of the iterations."""
        return self._iteration_timer


TPhaseState = PhaseState[TData, TStepOutput]


class State:
    """Parent State class which can contain up to 3 instances of PhaseState, for the 3 phases.
    Modified by the framework, read-only for the user.
    """

    def __init__(
        self,
        *,
        entry_point: EntryPoint,
        timer: Optional[TimerProtocol] = None,
        train_state: Optional[TPhaseState] = None,
        eval_state: Optional[TPhaseState] = None,
        predict_state: Optional[TPhaseState] = None,
    ) -> None:
        self._entry_point = entry_point
        self._timer = timer
        self._train_state = train_state
        self._eval_state = eval_state
        self._predict_state = predict_state
        self._should_stop: bool = False
        self._active_phase: ActivePhase = ActivePhase.TRAIN

    @property
    def entry_point(self) -> EntryPoint:
        """Entry point used to start loop execution. (One of FIT, TRAIN, EVALUATE, PREDICT)."""
        return self._entry_point

    @property
    def active_phase(self) -> ActivePhase:
        """Current active phase of the loop. (One of TRAIN, EVALUATE, PREDICT)."""
        return self._active_phase

    @property
    def timer(self) -> Optional[TimerProtocol]:
        """A :class:`~torchtnt.utils.TimerProtocol` object which can be used for debugging to record latencies of key events during loop execution."""
        return self._timer

    @property
    def train_state(self) -> Optional[TPhaseState]:
        """A :class:`~torchtnt.framework.state.PhaseState` object which contains meta information about the train phase."""
        return self._train_state

    @property
    def eval_state(self) -> Optional[TPhaseState]:
        """A :class:`~torchtnt.framework.state.PhaseState` object which contains meta information about the eval phase."""
        return self._eval_state

    @property
    def predict_state(self) -> Optional[TPhaseState]:
        """A :class:`~torchtnt.framework.state.PhaseState` object which contains meta information about the predict phase."""
        return self._predict_state

    @property
    def should_stop(self) -> bool:
        """Read-only property for whether to terminate the loop after the current step completes."""
        return self._should_stop

    def stop(self) -> None:
        """Signal to the loop to end after the current step completes."""
        _logger.warning("Received signal to stop")
        self._should_stop = True

    def active_phase_state(self) -> TPhaseState:
        """Returns the current active phase state."""
        if self._active_phase == ActivePhase.TRAIN:
            return none_throws(self._train_state)
        elif self._active_phase == ActivePhase.EVALUATE:
            return none_throws(self._eval_state)
        elif self._active_phase == ActivePhase.PREDICT:
            return none_throws(self._predict_state)
        else:
            raise ValueError(f"Invalid active phase: {self._active_phase}")
