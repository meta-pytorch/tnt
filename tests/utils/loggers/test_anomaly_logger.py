#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import unittest
from unittest.mock import call, MagicMock, patch

import torch
from torchtnt.utils.anomaly_evaluation import (
    IsNaNEvaluator,
    MetricAnomalyEvaluator,
    ThresholdEvaluator,
)
from torchtnt.utils.loggers.anomaly_logger import AnomalyLogger, TrackedMetric


class DummyEvaluator(MetricAnomalyEvaluator):
    def _evaluate_anomaly(self, value: float) -> bool:
        return True


class TestAnomalyLogger(unittest.TestCase):

    def test_init(self) -> None:
        tracked_metrics = [
            TrackedMetric(
                name="accuracy",
                anomaly_evaluators=[ThresholdEvaluator(min_val=0.5, max_val=0.9)],
            ),
            TrackedMetric(
                name="accuracy",
                anomaly_evaluators=[IsNaNEvaluator()],
            ),
            TrackedMetric(name="loss", anomaly_evaluators=[IsNaNEvaluator()]),
        ]

        warning_container = []
        with patch(
            "torchtnt.utils.loggers.anomaly_logger.logging.Logger.warning",
            side_effect=warning_container.append,
        ):
            logger = AnomalyLogger(
                tracked_metrics=tracked_metrics,
            )

        self.assertEqual(
            warning_container,
            ["Found multiple configs for metric 'accuracy'. Skipping."],
        )
        self.assertEqual(set(logger._tracked_metrics.keys()), {"loss"})

    @patch(
        "torchtnt.utils.loggers.anomaly_logger.AnomalyLogger.on_anomaly_detected",
    )
    def test_log(self, mock_on_anomaly_detected: MagicMock) -> None:
        logger = AnomalyLogger(
            tracked_metrics=[
                TrackedMetric(
                    name="accuracy",
                    anomaly_evaluators=[ThresholdEvaluator(min_val=0.5, max_val=0.9)],
                    warmup_steps=4,
                    evaluate_every_n_steps=2,
                )
            ]
        )

        # Log value that can't be resolved to a single numerical.
        warning_container = []
        with patch(
            "torchtnt.utils.loggers.anomaly_logger.logging.Logger.warning",
            side_effect=warning_container.append,
        ):
            logger.log(step=1, name="accuracy", data=torch.Tensor([0.5, 0.9]))

        self.assertEqual(
            warning_container,
            [
                "Error when extracting a single numerical value from the provided metric: Scalar tensor must contain a single item, 2 given."
            ],
        )
        mock_on_anomaly_detected.assert_called_once()

        # Log anomalous value during warmup: no-op
        mock_on_anomaly_detected.reset_mock()
        logger.log(step=4, name="accuracy", data=0.2)
        mock_on_anomaly_detected.assert_not_called()

        # Log anomalous value on non-evaluate step: no-op
        logger.log(step=5, name="accuracy", data=0.1)
        mock_on_anomaly_detected.assert_not_called()

        # Log metric that is not tracked: no-op
        mock_on_anomaly_detected.reset_mock()
        logger.log(step=6, name="loss", data=math.nan)
        mock_on_anomaly_detected.assert_not_called()

        # Log metric within threshold: no-op
        logger.log(step=6, name="accuracy", data=0.6)
        mock_on_anomaly_detected.assert_not_called()

        # Log metric outside threshold
        warning_container = []
        with patch(
            "torchtnt.utils.loggers.anomaly_logger.logging.Logger.warning",
            side_effect=warning_container.append,
        ):
            logger.log(step=8, name="accuracy", data=0.95)

        self.assertEqual(
            warning_container,
            [
                "Found anomaly in metric: accuracy, with value: 0.95, using evaluator: ThresholdEvaluator"
            ],
        )
        mock_on_anomaly_detected.assert_called_with("accuracy", 0.95, 8)

    @patch(
        "torchtnt.utils.loggers.anomaly_logger.AnomalyLogger.on_anomaly_detected",
    )
    def test_log_dict(self, mock_on_anomaly_detected: MagicMock) -> None:
        logger = AnomalyLogger(
            tracked_metrics=[
                TrackedMetric(
                    name="accuracy",
                    anomaly_evaluators=[ThresholdEvaluator(min_val=0.5, max_val=0.9)],
                ),
                TrackedMetric(
                    name="loss",
                    anomaly_evaluators=[IsNaNEvaluator()],
                ),
                TrackedMetric(
                    name="f1_score",
                    anomaly_evaluators=[
                        IsNaNEvaluator(),
                        ThresholdEvaluator(min_val=0.2),
                    ],
                ),
            ]
        )

        warning_container = []
        with patch(
            "torchtnt.utils.loggers.anomaly_logger.logging.Logger.warning",
            side_effect=warning_container.append,
        ):
            logger.log_dict(
                step=1,
                payload={
                    "loss": math.nan,
                    "accuracy": 0.63,
                    "precision": 0.7,
                    "f1_score": 0.05,
                },
            )

        self.assertEqual(
            set(warning_container),
            {
                "Found anomaly in metric: f1_score, with value: 0.05, using evaluator: ThresholdEvaluator",
                "Found anomaly in metric: loss, with value: nan, using evaluator: IsNaNEvaluator",
            },
        )

        expected_anomaly_callback_calls = [
            call("f1_score", 0.05, 1),
            call("loss", math.nan, 1),
        ]
        mock_on_anomaly_detected.assert_has_calls(
            expected_anomaly_callback_calls, any_order=True
        )

    @patch(
        "torchtnt.utils.loggers.anomaly_logger.AnomalyLogger.on_anomaly_detected",
        side_effect=Exception("test exception"),
    )
    def test_on_anomaly_callback_exception(self, _) -> None:
        logger = AnomalyLogger(
            tracked_metrics=[
                TrackedMetric(
                    name="accuracy",
                    anomaly_evaluators=[ThresholdEvaluator(min_val=0.5, max_val=0.9)],
                ),
            ]
        )

        warning_container = []
        with patch(
            "torchtnt.utils.loggers.anomaly_logger.logging.Logger.warning",
            side_effect=warning_container.append,
        ):
            logger.log(step=1, name="accuracy", data=0.95)

        self.assertEqual(
            warning_container,
            [
                "Found anomaly in metric: accuracy, with value: 0.95, using evaluator: ThresholdEvaluator",
                "Exception when calling on_anomaly_hook: test exception",
            ],
        )
