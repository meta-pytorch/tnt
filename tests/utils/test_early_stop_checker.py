#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, Literal

import torch
from torchtnt.utils.early_stop_checker import EarlyStopChecker


class EarlyStopCheckerTest(unittest.TestCase):
    def test_reused_metric_tensor(self) -> None:
        for mode, values in (("min", [4.0, 2.0, 2.0]), ("max", [2.0, 4.0, 4.0])):
            for shape in ((), (1,), (1, 1)):
                with self.subTest(mode=mode, shape=shape):
                    checker = EarlyStopChecker(
                        cast(Literal["min", "max"], mode), patience=1
                    )
                    metric = torch.empty(shape)
                    decisions = []
                    for value in values:
                        metric.fill_(value)
                        decisions.append(checker.check(metric))
                    self.assertEqual(decisions, [False, False, True])

    def test_best_value_is_detached_snapshot(self) -> None:
        checker = EarlyStopChecker("min", patience=1)
        metric = torch.tensor(4.0, requires_grad=True).square()
        self.assertFalse(checker.check(metric))
        best_value = checker.state_dict()["best_value"]

        with torch.no_grad():
            metric.zero_()

        self.assertEqual(best_value.item(), 16.0)
        self.assertFalse(best_value.requires_grad)
        self.assertIsNone(best_value.grad_fn)

    def test_reused_loss_buffer_in_training_loop(self) -> None:
        weight = torch.nn.Parameter(torch.tensor(4.0))
        optimizer = torch.optim.SGD([weight], lr=0.25)
        checker = EarlyStopChecker("min", patience=1)
        loss_buffer = torch.zeros(())
        losses = []

        for _ in range(6):
            optimizer.zero_grad()
            loss = weight.square()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss_buffer.copy_(weight.square())
            losses.append(loss_buffer.item())
            if checker.check(loss_buffer):
                break

        # SGD halves the weight each step, so every loss strictly improves.
        self.assertEqual(losses, [4.0, 1.0, 0.25, 0.0625, 0.015625, 0.00390625])
        self.assertEqual(weight.item(), 0.0625)

    def test_early_stop_patience(self) -> None:
        # Loss does not decrease beyond 0.25
        losses = [0.4, 0.3, 0.28, 0.25, 0.26, 0.25]
        es1 = EarlyStopChecker("min", 3)
        es2 = EarlyStopChecker("min", 4)

        for loss in losses:
            should_stop = es1.check(torch.tensor(loss))
            self.assertFalse(should_stop)
            should_stop = es2.check(torch.tensor(loss))
            self.assertFalse(should_stop)

        # Patience should run out
        should_stop = es1.check(torch.tensor(0.25))
        self.assertTrue(should_stop)

        # es2 has more patience than es1
        should_stop = es2.check(torch.tensor(0.25))
        self.assertFalse(should_stop)
        should_stop = es2.check(torch.tensor(0.25))
        self.assertTrue(should_stop)

    def test_early_stop_float(self) -> None:

        # Same as previous test but with floats
        losses = [0.4, 0.3, 0.28, 0.25, 0.26, 0.25]
        es1 = EarlyStopChecker("min", 3)
        es2 = EarlyStopChecker("min", 4)

        for loss in losses:
            should_stop = es1.check(loss)
            self.assertFalse(should_stop)
            should_stop = es2.check(loss)
            self.assertFalse(should_stop)

        # Patience should run out
        should_stop = es1.check(0.25)
        self.assertTrue(should_stop)

        # es2 has more patience than es1
        should_stop = es2.check(0.25)
        self.assertFalse(should_stop)
        should_stop = es2.check(0.25)
        self.assertTrue(should_stop)

        # floats should be converted to tensors
        self.assertTrue(torch.is_tensor(es2._best_value))

    def test_early_stop_min_delta(self) -> None:

        # Loss decreases beyond 0.25 but not more than min_delta
        losses = [0.4, 0.38, 0.31, 0.25, 0.27, 0.24]
        es1 = EarlyStopChecker("min", 3, min_delta=0.05)
        es2 = EarlyStopChecker("min", 4, min_delta=0.05)

        for loss in losses:
            should_stop = es1.check(torch.tensor(loss))
            self.assertFalse(should_stop)
            should_stop = es2.check(torch.tensor(loss))
            self.assertFalse(should_stop)

        # Patience should run out
        should_stop = es1.check(torch.tensor(0.25))
        self.assertTrue(should_stop)

        # es2 has more patience than es1
        should_stop = es2.check(torch.tensor(0.25))
        self.assertFalse(should_stop)
        should_stop = es2.check(torch.tensor(0.26))
        self.assertTrue(should_stop)

    def test_early_stop_max_mode(self) -> None:

        # Loss increases beyond 0.38 but not more than min_delta
        losses = [0.25, 0.3, 0.32, 0.38, 0.39, 0.41]
        es1 = EarlyStopChecker("max", 3, min_delta=0.05)
        es2 = EarlyStopChecker("max", 4, min_delta=0.05)

        for loss in losses:
            should_stop = es1.check(torch.tensor(loss))
            self.assertFalse(should_stop)
            should_stop = es2.check(torch.tensor(loss))
            self.assertFalse(should_stop)

        # Patience should run out
        should_stop = es1.check(torch.tensor(0.39))
        self.assertTrue(should_stop)

        # es2 has more patience than es1
        should_stop = es2.check(torch.tensor(0.39))
        self.assertFalse(should_stop)
        should_stop = es2.check(torch.tensor(0.35))
        self.assertTrue(should_stop)

    def test_early_stop_check_finite(self) -> None:

        losses = [0.4, 0.3, 0.28, 0.25]
        es1 = EarlyStopChecker("min", 3, check_finite=True)
        es2 = EarlyStopChecker("min", 3, check_finite=True)
        es3 = EarlyStopChecker("min", 3, check_finite=True)

        # Make sure check_finite does not break normal usage
        for loss in losses:
            should_stop = es1.check(torch.tensor(loss))
            self.assertFalse(should_stop)
            should_stop = es2.check(torch.tensor(loss))
            self.assertFalse(should_stop)
            should_stop = es3.check(torch.tensor(loss))
            self.assertFalse(should_stop)

        # Check positive inf, negative inf, and NAN
        should_stop = es1.check(torch.tensor(float("-inf")))
        self.assertTrue(should_stop)
        should_stop = es2.check(torch.tensor(float("-inf")))
        self.assertTrue(should_stop)
        should_stop = es3.check(torch.tensor(float("nan")))
        self.assertTrue(should_stop)

    def test_early_stop_stopping_threshold_min(self) -> None:

        # Loss does not reach threshold, patience is high
        losses = [0.4, 0.3, 0.28, 0.25, 0.26, 0.25]
        es1 = EarlyStopChecker("min", 6, stopping_threshold=0.24)
        es2 = EarlyStopChecker("min", 6, stopping_threshold=0.24)

        for loss in losses:
            should_stop = es1.check(torch.tensor(loss))
            self.assertFalse(should_stop)
            should_stop = es2.check(torch.tensor(loss))
            self.assertFalse(should_stop)

        # Loss drops below threshold when patience remaining = 4
        should_stop = es1.check(torch.tensor(0.23))
        self.assertTrue(should_stop)

        # Execution continues with patience remaining = 4
        should_stop = es2.check(torch.tensor(0.24))
        self.assertFalse(should_stop)
        # Loss drops below threshold with patience remaining = 3
        should_stop = es2.check(torch.tensor(0.23))
        self.assertTrue(should_stop)

    def test_early_stop_stopping_threshold_max(self) -> None:

        # Loss does not reach threshold, patience is high
        losses = [0.2, 0.3, 0.38, 0.4, 0.39, 0.4]
        es1 = EarlyStopChecker("max", 6, stopping_threshold=0.41)
        es2 = EarlyStopChecker("max", 6, stopping_threshold=0.41)

        for loss in losses:
            should_stop = es1.check(torch.tensor(loss))
            self.assertFalse(should_stop)
            should_stop = es2.check(torch.tensor(loss))
            self.assertFalse(should_stop)

        # Loss exceeds threshold when patience remaining = 4
        should_stop = es1.check(torch.tensor(0.42))
        self.assertTrue(should_stop)

        # Execution continues with patience remaining = 4
        should_stop = es2.check(torch.tensor(0.41))
        self.assertFalse(should_stop)
        # Loss exceeds threshold with patience remaining = 3
        should_stop = es2.check(torch.tensor(0.42))
        self.assertTrue(should_stop)

    def test_early_stop_improvement_threshold_rel_min(self) -> None:

        # Loss does not exceed threshold, patience is high
        losses = [0.4, 0.39, 0.31, 0.25, 0.27, 0.24]
        es1 = EarlyStopChecker("min", 3, min_delta=0.05, threshold_mode="rel")
        es2 = EarlyStopChecker("min", 4, min_delta=0.05, threshold_mode="rel")

        for loss in losses:
            should_stop = es1.check(torch.tensor(loss))
            self.assertFalse(should_stop)
            should_stop = es2.check(torch.tensor(loss))
            self.assertFalse(should_stop)

        # Patience should run out
        should_stop = es1.check(torch.tensor(0.25))
        self.assertTrue(should_stop)

        # es2 has more patience than es1
        should_stop = es2.check(torch.tensor(0.25))
        self.assertFalse(should_stop)
        should_stop = es2.check(torch.tensor(0.24))
        self.assertTrue(should_stop)

    def test_early_stop_improvement_threshold_rel_max(self) -> None:

        # Loss does not exceed threshold, patience is high
        losses = [0.25, 0.26, 0.32, 0.38, 0.39, 0.41]
        es1 = EarlyStopChecker("max", 3, min_delta=0.1, threshold_mode="rel")
        es2 = EarlyStopChecker("max", 4, min_delta=0.1, threshold_mode="rel")

        for loss in losses:
            should_stop = es1.check(torch.tensor(loss))
            self.assertFalse(should_stop)
            should_stop = es2.check(torch.tensor(loss))
            self.assertFalse(should_stop)

        # Patience should run out
        should_stop = es1.check(torch.tensor(0.41))
        self.assertTrue(should_stop)

        # es2 has more patience than es1
        should_stop = es2.check(torch.tensor(0.40))
        self.assertFalse(should_stop)
        should_stop = es2.check(torch.tensor(0.39))
        self.assertTrue(should_stop)

    def test_early_stop_divergence_threshold_min(self) -> None:

        # Loss does not exceed threshold, patience is high
        losses = [0.4, 0.3, 0.28, 0.25, 0.25, 0.3]
        es1 = EarlyStopChecker("min", 6, divergence_threshold=0.5)
        es2 = EarlyStopChecker("min", 6, divergence_threshold=0.5)

        for loss in losses:
            should_stop = es1.check(torch.tensor(loss))
            self.assertFalse(should_stop)
            should_stop = es2.check(torch.tensor(loss))
            self.assertFalse(should_stop)

        # Loss exceeds threshold when patience remaining = 4
        should_stop = es1.check(torch.tensor(0.51))
        self.assertTrue(should_stop)

        # Execution continues with patience remaining = 4
        should_stop = es2.check(torch.tensor(0.5))
        self.assertFalse(should_stop)
        # Loss exceeds threshold with patience remaining = 3
        should_stop = es2.check(torch.tensor(0.51))
        self.assertTrue(should_stop)

    def test_early_stop_divergence_threshold_max(self) -> None:

        # Loss does not exceed threshold, patience is high
        losses = [0.2, 0.3, 0.38, 0.4, 0.3, 0.1]
        es1 = EarlyStopChecker("max", 6, divergence_threshold=-0.05)
        es2 = EarlyStopChecker("max", 6, divergence_threshold=-0.05)

        for loss in losses:
            should_stop = es1.check(torch.tensor(loss))
            self.assertFalse(should_stop)
            should_stop = es2.check(torch.tensor(loss))
            self.assertFalse(should_stop)

        # Loss drops below threshold when patience remaining = 4
        should_stop = es1.check(torch.tensor(-0.06))
        self.assertTrue(should_stop)

        # Execution continues with patience remaining = 4
        should_stop = es2.check(torch.tensor(-0.03))
        self.assertFalse(should_stop)
        # Loss drops below threshold with patience remaining = 3
        should_stop = es2.check(torch.tensor(-0.06))
        self.assertTrue(should_stop)

    def test_early_stop_reset(self) -> None:

        losses = [0.4, 0.3, 0.28, 0.25, 0.26, 0.25]
        es1 = EarlyStopChecker("min", 3)
        es2 = EarlyStopChecker("min", 3)

        for loss in losses:
            should_stop = es1.check(torch.tensor(loss))
            self.assertFalse(should_stop)
            should_stop = es2.check(torch.tensor(loss))
            self.assertFalse(should_stop)

        # Patience should run out
        should_stop = es1.check(torch.tensor(0.25))
        self.assertTrue(should_stop)

        # Reset should prevent patience from running out
        es2.reset()
        should_stop = es2.check(torch.tensor(0.25))
        self.assertFalse(should_stop)

        # Reset should allow loop to run again
        es2.reset()
        for loss in losses:
            should_stop = es2.check(torch.tensor(loss))
            self.assertFalse(should_stop)
        should_stop = es2.check(torch.tensor(0.25))
        self.assertTrue(should_stop)

    def test_early_stop_state_dict(self) -> None:

        losses = [0.4, 0.3, 0.28, 0.25, 0.26, 0.25]
        es1 = EarlyStopChecker("min", 3)
        es2 = EarlyStopChecker("min", 3)

        # Only update wait count and best_value on es1
        for loss in losses:
            should_stop = es1.check(torch.tensor(loss))
            self.assertFalse(should_stop)

        # Verify contents of state_dict
        state_dict = es1.state_dict()
        self.assertEqual(state_dict["patience_count"], 2)
        self.assertEqual(state_dict["best_value"].item(), 0.25)

        # es2 should behave like es1 after loading state
        es2.load_state_dict(state_dict)

        should_stop = es1.check(torch.tensor(0.25))
        self.assertTrue(should_stop)
        should_stop = es2.check(torch.tensor(0.25))
        self.assertTrue(should_stop)

    def test_early_stop_invalid_mode(self) -> None:

        # Check for invalid mode
        with self.assertRaisesRegex(ValueError, "Got `invalid`"):
            # pyre-fixme[6]: For 1st argument expected
            #  `Union[typing_extensions.Literal['max'],
            #  typing_extensions.Literal['min']]` but got
            #  `typing_extensions.Literal['invalid']`.
            EarlyStopChecker("invalid", 3)

    def test_early_stop_invalid_min_delta(self) -> None:

        # Check for invalid min_delta
        with self.assertRaisesRegex(
            ValueError, "`min_delta` must be greater than or equal to 0. Got -1"
        ):
            EarlyStopChecker("min", 3, min_delta=-1)

    def test_early_stop_invalid_input_tensor(self) -> None:

        es = EarlyStopChecker("min", 3)
        # Check for invalid input tensor
        with self.assertRaisesRegex(ValueError, "number of elements = 3"):
            es.check(torch.ones(3))

    def test_early_stop_properties(self) -> None:

        es1 = EarlyStopChecker("min", 3, 0.3, False, "rel", 2.2, 8.5)
        es2 = EarlyStopChecker("max", 5)

        self.assertEqual(es1.mode, "min")
        self.assertEqual(es1.patience, 3)
        self.assertEqual(es1.min_delta, -0.3)
        self.assertFalse(es1.check_finite)
        self.assertEqual(es1.threshold_mode, "rel")
        self.assertEqual(es1.stopping_threshold, 2.2)
        self.assertEqual(es1.divergence_threshold, 8.5)
        self.assertEqual(es1._mode_func, torch.lt)
        self.assertEqual(es1._mode_char, "<")

        self.assertEqual(es2.mode, "max")
        self.assertEqual(es2.patience, 5)
        self.assertEqual(es2.min_delta, 0.0)
        self.assertTrue(es2.check_finite)
        self.assertEqual(es2.threshold_mode, "abs")
        self.assertIsNone(es2.stopping_threshold)
        self.assertIsNone(es2.divergence_threshold)
        self.assertEqual(es2._mode_func, torch.gt)
        self.assertEqual(es2._mode_char, ">")

    def test_check_input_validation(self) -> None:
        es = EarlyStopChecker("min", 3)
        self.assertEqual(es._best_value.size(), torch.Size([1]))
        self.assertEqual(es._best_value.dtype, torch.float32)

        es.check(5)
        self.assertEqual(es._best_value, torch.tensor([5.0]))
        self.assertEqual(es._best_value.size(), torch.Size([1]))
        self.assertEqual(es._best_value.dtype, torch.float32)

        es.check(4.0)
        self.assertEqual(es._best_value, torch.tensor([4.0]))
        self.assertEqual(es._best_value.size(), torch.Size([1]))
        self.assertEqual(es._best_value.dtype, torch.float32)

        with self.assertRaisesRegex(
            ValueError,
            "Expected tensor with only 1 element, but input has number of elements = 2",
        ):
            es.check(torch.tensor([3, 2]))

        es.check(torch.tensor([1]))
        self.assertEqual(es._best_value, torch.tensor([1.0]))
        self.assertEqual(es._best_value.size(), torch.Size([1]))
        self.assertEqual(es._best_value.dtype, torch.float32)

        es.check(torch.tensor(0))
        self.assertEqual(es._best_value, torch.tensor([0.0]))
        self.assertEqual(es._best_value.size(), torch.Size([1]))
        self.assertEqual(es._best_value.dtype, torch.float32)
