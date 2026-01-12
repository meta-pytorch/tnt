#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Set

from torchtnt.utils.anomaly_evaluation import MetricAnomalyEvaluator
from torchtnt.utils.loggers.logger import MetricLogger, Scalar
from torchtnt.utils.loggers.utils import scalar_to_float

_logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class TrackedMetric:
    """
    Specify a metric that will be tracked and evaluated for anomalies. Only metrics that can be resolved to a single numerical
    value are supported. If a Tensor or numpy array are passed, they must contain a single numerical value to be used.

    Args:
        name: Name of the metric.
        anomaly_evaluator: Evaluators to use for anomaly detection. Should implement the :py:class:`~torchtnt.utils.loggers.metric_anomaly_logger.MetricAnomalyEvaluator`
            interface. Current options are :py:class:`~torchtnt.utils.loggers.metric_anomaly_logger.Threshold` and :py:class:`~torchtnt.utils.loggers.IsNaN
        warmup_steps: Optional number of steps to wait before evaluating metric anomalies. During this period, only the evaluator's `update` method will be called.
            Default value is 0.
        evaluate_every_n_steps: Interval at which to evaluate anomalies. Default value is 1, so they are evaluated every step.
    """

    name: str
    anomaly_evaluators: List[MetricAnomalyEvaluator]
    warmup_steps: int = 0
    evaluate_every_n_steps: int = 1


class AnomalyLogger(MetricLogger):
    """
    Logger that evaluates if metric values to check for anomalies. If an anomaly is detected, a warning is logged and an
    optional callback is called. This is useful to execute side effects like sending notifications, writing to a database, etc.

    Metrics can be configured using the :py:meth:`~torchtnt.utils.loggers.metric_anomaly_logger.TrackedMetric`
    dataclass. They will not be logged if they are within the acceptable range of values. It is possible to pair this up with
    another logger via subclassing or composition.

    Example:

    .. code-block:: python

        from torchtnt.utils import ThresholdEvaluator
        from torchtnt.utils.loggers import MetricAnomalyLogger

        # Define a dummy custom logger that logs to a file. If no side effects are needed,
        # we can use MetricAnomalyLogger directly.
        class AnomalyRecorder(AnomalyLogger):
            def on_anomaly_detected(self, name: str, value: Scalar, step: int) -> None:
                # Function to log metric anomalies to a text file.
                with open("anomaly_record.txt", "a") as anomaly_record:
                    anomaly_record.write(f"{name=}, {value=}, {step=}")

        logger = AnomalyRecorder(
            tracked_metrics=[
                TrackedMetric(
                    name="accuracy",
                    anomaly_evaluator=[ThresholdEvaluator(min_val=0.5, max_val=0.95)],
                    warmup_steps=1,
                )
            ]
        )

        # Calling within the warmup period will be no-op
        logger.log(step=1, name="accuracy", data=0.9734)

        # This will log the warning and write to the file anomaly_record.txt
        logger.log(step=2, name="accuracy", data=0.982378)

        # This will be a no-op since the value is within the acceptable range
        logger.log(step=3, name="loss", data=0.5294)
    """

    def __init__(self, tracked_metrics: Optional[List[TrackedMetric]] = None) -> None:
        """
        Args:
            tracked_metrics: List of metrics to track and evaluate for anomalies. If not provided, no metrics will be tracked.
                Note that a single configuration should be passed for one metric. If it is duplicated, the metric will be ignored
                for anomaly detection.
        """
        self._tracked_metrics: Dict[str, TrackedMetric] = {}
        if not tracked_metrics:
            return

        duplicated: Set[str] = set()
        for metric in tracked_metrics:
            if metric.name in self._tracked_metrics or metric.name in duplicated:
                _logger.warning(
                    f"Found multiple configs for metric '{metric.name}'. Skipping."
                )
                del self._tracked_metrics[metric.name]
                duplicated.add(metric.name)
                continue

            self._tracked_metrics[metric.name] = metric

        _logger.info(
            f"Started tracking anomalies for the following metrics: {self._tracked_metrics.keys()}"
        )

    def _maybe_evaluate_and_log(self, name: str, data: Scalar, step: int) -> None:
        metric_config = self._tracked_metrics.get(name)
        if not metric_config:
            return

        try:
            data_f = scalar_to_float(data)
        except ValueError as exc:
            _logger.warning(
                f"Error when extracting a single numerical value from the provided metric: {exc}"
            )
            self.on_anomaly_detected(name, data, step)
            return

        for evaluator in metric_config.anomaly_evaluators:
            evaluator.update(data_f)

            if (
                step <= metric_config.warmup_steps
                or step % metric_config.evaluate_every_n_steps != 0
            ):
                continue

            if not evaluator.is_anomaly():
                continue

            _logger.warning(
                f"Found anomaly in metric: {name}, with value: {data}, "
                f"using evaluator: {evaluator.__class__.__name__}"
            )

            try:
                self.on_anomaly_detected(name, data_f, step)
            except Exception as e:
                _logger.warning(f"Exception when calling on_anomaly_hook: {str(e)}")

    def log(
        self,
        name: str,
        data: Scalar,
        step: int,
    ) -> None:
        """
        If `name` matches a registered metric, for each evaluator in the metric's `anomaly_evaluators`
        list, the `update` method will always be called.Then, will determine if metric should be evaluated
        at the current `step` based on the metric's `warmup_steps` and `evaluate_every_n_step` config.
        If so, the `is_anomaly` method will be called.

        If the metric value is determined to be anomalous by any configured evaluator, the anomaly will be
        logged, and the `on_anomaly_detected` callback will be executed. This will also happen if the input
        provided in `data` cannot be resolved to a single `float` value.
        """
        self._maybe_evaluate_and_log(name, data, step)

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        """
        The same logic described in the `log` method will be applied to every metric in the `payload` mapping.
        """
        for metric, data in payload.items():
            self._maybe_evaluate_and_log(metric, data, step)

    def on_anomaly_detected(self, name: str, data: Scalar, step: int) -> None:
        """
        Callback method to be executed when an anomaly in a tracked metric is detected.
        Override this to execute custom side effects. Note that exceptions in this method
        will be handled.

        Args:
            name: Name of the metric with the anomalous value.
            data: Value of the metric.
            step: Step value to record.
        """
        pass
