#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
from abc import ABC, abstractmethod
from math import inf

_logger: logging.Logger = logging.getLogger(__name__)


class MetricAnomalyEvaluator(ABC):
    """
    Abstract base class for metric anomaly evaluators. An evaluator specifies the logic to determine that
    a particular metric value is anomalous. To implement a custom method, create a subclass and implement
    the following methods:
    - :py:meth:`~torchtnt.utils.loggers.metric_anomaly_logger.MetricAnomalyEvaluator.update` should receive
        the metric value and update the internal state. This is specially useful for algorithms that require
        storing some previous values, moving averages, etc.
    - :py:meth:`~torchtnt.utils.loggers.metric_anomaly_logger.MetricAnomalyEvaluator.is_anomaly` determines
        whether the current metric state is anomalous.

    Likely there are some warm-up steps before the metric is stable and can be checked against anomalies, so
    the separation of state update and actual detection provides this flexibility.
    """

    @abstractmethod
    def update(self, value: float) -> None:
        """
        Update the internal state with the given metric value. This should not determine anomalies itself, but
        only aggregate the current value according to the anomaly detection algorithm.

        Note:: If no aggregation is required, this method can store the value directly, to be used in `is_anomaly`.

        Args:
            value: Metric value
        """
        pass

    @abstractmethod
    def is_anomaly(self) -> bool:
        """
        Determine whether the current metric state is anomalous. This should be overridden with custom logic related to
        an anomaly detection algorithm.
        """
        pass


class ThresholdEvaluator(MetricAnomalyEvaluator):
    """
    Evaluates whether a metric value is anomalous based on a predefined threshold.
    """

    def __init__(
        self,
        *,
        min_val: float = -inf,
        max_val: float = inf,
    ) -> None:
        """
        Args:
            min_val: Minimum allowed value. Default value is -inf.
            max_val: Maximum allowed value. Default value is inf.
            warmup_steps: Number of steps to ignore before evaluating anomalies. Default value is 0.
            evaluate_every_n_steps: Step interval to wait in between anomaly evaluations. Default value is 1.
        """
        self.min_val = min_val
        self.max_val = max_val
        self.curr_val: float = min_val

    def update(self, value: float) -> None:
        self.curr_val = value

    def is_anomaly(self) -> bool:
        return not self.min_val <= self.curr_val <= self.max_val


class IsNaNEvaluator(MetricAnomalyEvaluator):
    """
    Evaluates whether a metric value is NaN.
    """

    def __init__(
        self,
    ) -> None:
        self.curr_val: float = 0

    def update(self, value: float) -> None:
        self.curr_val = value

    def is_anomaly(self) -> bool:
        return math.isnan(self.curr_val)
