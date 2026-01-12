#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Dict
from unittest.mock import patch

import torch
from torch import nn
from torchtnt.framework._test_utils import DummyAutoUnit
from torchtnt.framework.unit import AppStateMixin
from torchtnt.utils.env import init_from_env
from torchtnt.utils.lr_scheduler import TLRScheduler
from torchtnt.utils.prepare_module import FSDP2OptimizerWrapper, FSDPOptimizerWrapper
from torchtnt.utils.stateful import MultiStateful


class Dummy(AppStateMixin):
    def __init__(self) -> None:
        super().__init__()
        self.module_a = nn.Linear(1, 1)
        self.loss_fn_b = nn.CrossEntropyLoss()
        self.optimizer_c = torch.optim.SGD(self.module_a.parameters(), lr=0.1)
        self.lr_scheduler_d = torch.optim.lr_scheduler.StepLR(
            self.optimizer_c, step_size=30, gamma=0.1
        )
        self.grad_scaler_e = torch.amp.GradScaler("cuda")
        self.optimizer_class_f = torch.optim.SGD


class AppStateMixinTest(unittest.TestCase):
    def test_tracked_modules(self) -> None:
        """
        Test setting, getting, and deleting tracked_modules
        """

        my_unit = Dummy()

        # assert that the attributes are stored in tracked_modules
        self.assertEqual(my_unit.tracked_modules()["module_a"], my_unit.module_a)
        self.assertEqual(my_unit.tracked_modules()["loss_fn_b"], my_unit.loss_fn_b)

        # delete the attributes
        # pyre-fixme[8]: Attribute has type `Linear`; used as `None`.
        my_unit.module_a = None
        # pyre-fixme[8]: Attribute has type `CrossEntropyLoss`; used as `None`.
        my_unit.loss_fn_b = None

        # the attributes should be removed from tracked_modules
        self.assertFalse("module_a" in my_unit.tracked_modules())
        self.assertFalse("loss_fn_b" in my_unit.tracked_modules())

    def test_tracked_optimizers(self) -> None:
        """
        Test setting, getting, and deleting tracked_optimizers
        """
        my_unit = Dummy()

        # assert that the attribute is stored in tracked_optimizers
        self.assertEqual(
            my_unit.tracked_optimizers()["optimizer_c"], my_unit.optimizer_c
        )

        # delete the attribute
        # pyre-fixme[8]: Attribute has type `SGD`; used as `None`.
        my_unit.optimizer_c = None

        # the attribute should be removed from tracked_optimizers
        self.assertFalse("optimizer_c" in my_unit.tracked_optimizers())

    def test_tracked_lr_schedulers(self) -> None:
        """
        Test setting, getting, and deleting tracked_lr_schedulers
        """

        my_unit = Dummy()

        # assert that the attribute is stored in tracked_lr_schedulers
        self.assertEqual(
            my_unit.tracked_lr_schedulers()["lr_scheduler_d"], my_unit.lr_scheduler_d
        )

        # delete the attribute
        # pyre-fixme[8]: Attribute has type `StepLR`; used as `None`.
        my_unit.lr_scheduler_d = None

        # the attribute should be removed from tracked_lr_schedulers
        self.assertFalse("lr_scheduler_d" in my_unit.tracked_lr_schedulers())

    def test_miscellaneous_stateful(self) -> None:
        """
        Test setting and getting miscellaneous stateful objects
        """

        my_unit = Dummy()

        # assert that the grad scaler is stored in the app_state
        self.assertEqual(my_unit.app_state()["grad_scaler_e"], my_unit.grad_scaler_e)

        # assert that only stateful class objects are being tracked
        self.assertFalse("optimizer_class_f" in my_unit.tracked_misc_statefuls())

        multi_stateful = MultiStateful(my_unit.tracked_misc_statefuls())
        try:
            _ = multi_stateful.state_dict()
        except TypeError:
            self.fail("Not able to get the state dict from my_unit.")

        # delete the attribute
        # pyre-fixme[8]: Attribute has type `GradScaler`; used as `None`.
        my_unit.grad_scaler_e = None

        # the attribute should be removed from tracked_misc_statefuls
        self.assertFalse("grad_scaler_e" in my_unit.tracked_misc_statefuls())

    def test_app_state(self) -> None:
        """
        Test the app_state method
        """

        my_unit = Dummy()

        # the attributes should be in app_state
        for key in (
            "module_a",
            "loss_fn_b",
            "optimizer_c",
            "lr_scheduler_d",
            "grad_scaler_e",
        ):
            self.assertTrue(key in my_unit.app_state())

        # delete the attributes
        # pyre-fixme[8]: Attribute has type `Linear`; used as `None`.
        my_unit.module_a = None
        # pyre-fixme[8]: Attribute has type `CrossEntropyLoss`; used as `None`.
        my_unit.loss_fn_b = None
        # pyre-fixme[8]: Attribute has type `SGD`; used as `None`.
        my_unit.optimizer_c = None
        # pyre-fixme[8]: Attribute has type `StepLR`; used as `None`.
        my_unit.lr_scheduler_d = None
        # pyre-fixme[8]: Attribute has type `GradScaler`; used as `None`.
        my_unit.grad_scaler_e = None

        # the attributes should no longer be in app_state
        for key in (
            "module_a",
            "loss_fn_b",
            "optimizer_c",
            "lr_scheduler_d",
            "grad_scaler_e",
        ):
            self.assertFalse(key in my_unit.app_state())

    def test_reassigning_attributes(self) -> None:
        """
        Test reassigning attributes to a different type
        """

        my_unit = Dummy()

        # create new objects
        module_e = nn.Linear(1, 1)
        loss_fn_f = nn.CrossEntropyLoss()
        optimizer_g = torch.optim.SGD(module_e.parameters(), lr=0.1)
        lr_scheduler_h = torch.optim.lr_scheduler.StepLR(
            optimizer_g, step_size=30, gamma=0.1
        )

        # reassigning module_a to be an optimizer should work
        self.assertTrue("module_a" in my_unit.tracked_modules())
        # pyre-fixme[8]: Attribute has type `Linear`; used as `SGD`.
        my_unit.module_a = optimizer_g
        self.assertTrue("module_a" not in my_unit.tracked_modules())
        self.assertTrue("module_a" in my_unit.tracked_optimizers())

        # reassigning optimizer_c to be an lr_scheduler should work
        self.assertTrue("optimizer_c" in my_unit.tracked_optimizers())
        # pyre-fixme[8]: Attribute has type `SGD`; used as `StepLR`.
        my_unit.optimizer_c = lr_scheduler_h
        self.assertTrue("optimizer_c" not in my_unit.tracked_optimizers())
        self.assertTrue("optimizer_c" in my_unit.tracked_lr_schedulers())

        # reassigning lr_scheduler_d to be a nn.module should work
        self.assertTrue("lr_scheduler_d" in my_unit.tracked_lr_schedulers())
        # pyre-fixme[8]: Attribute has type `StepLR`; used as `CrossEntropyLoss`.
        my_unit.lr_scheduler_d = loss_fn_f
        self.assertTrue("lr_scheduler_d" not in my_unit.tracked_lr_schedulers())
        self.assertTrue("lr_scheduler_d" in my_unit.tracked_modules())

        self.assertTrue("grad_scaler_e" in my_unit.tracked_misc_statefuls())
        # pyre-fixme[8]: Attribute has type `GradScaler`; used as `CrossEntropyLoss`.
        my_unit.grad_scaler_e = loss_fn_f
        self.assertTrue("grad_scaler_e" not in my_unit.tracked_misc_statefuls())
        self.assertTrue("grad_scaler_e" in my_unit.tracked_modules())

    def test_app_state_overload(self) -> None:
        class Override(AppStateMixin):
            def __init__(self) -> None:
                super().__init__()
                self.module_a = nn.Linear(1, 1)
                self.loss_fn_b = nn.CrossEntropyLoss()
                self.optimizer_c = torch.optim.SGD(self.module_a.parameters(), lr=0.1)
                self.optimizer_placeholder = torch.optim.SGD(
                    self.module_a.parameters(), lr=0.2
                )
                self.lr_scheduler_d = torch.optim.lr_scheduler.StepLR(
                    self.optimizer_c, step_size=30, gamma=0.1
                )
                self.lr_2 = torch.optim.lr_scheduler.StepLR(
                    self.optimizer_placeholder, step_size=50, gamma=0.3
                )
                self.grad_scaler_e = torch.amp.GradScaler("cuda")

            def tracked_modules(self) -> Dict[str, nn.Module]:
                ret = super().tracked_modules()
                ret["another_module"] = nn.Linear(1, 1)
                return ret

            def tracked_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
                return {"optimizer_c": self.optimizer_c}

            def tracked_lr_schedulers(
                self,
            ) -> Dict[str, TLRScheduler]:
                return {"lr_2": self.lr_2}

            def tracked_misc_statefuls(self) -> Dict[str, Any]:
                ret = super().tracked_misc_statefuls()
                ret["another_scaler"] = torch.amp.GradScaler("cuda")
                return ret

        o = Override()
        app_state = o.app_state()
        self.assertIn("module_a", app_state)
        self.assertIn("loss_fn_b", app_state)
        # from overridden tracked_modules
        self.assertIn("another_module", app_state)

        self.assertIn("optimizer_c", app_state)
        self.assertNotIn("optimizer_placeholder", app_state)

        self.assertIn("lr_2", app_state)
        self.assertNotIn("lr_scheduler_d", app_state)
        self.assertIn("grad_scaler_e", app_state)
        self.assertIn("another_scaler", app_state)

    def test_construct_tracked_optimizers_and_schedulers(self) -> None:
        device = init_from_env()
        module = torch.nn.Linear(10, 10)

        auto_unit = DummyAutoUnit(module=module)
        auto_unit.module2 = torch.nn.Linear(10, 10).to(device)
        auto_unit.optim2 = torch.optim.Adam(auto_unit.module2.parameters())

        with patch(
            "torchtnt.framework.unit._is_fsdp_module", side_effect=lambda m: m == module
        ):
            result = auto_unit._construct_tracked_optimizers_and_schedulers()

        self.assertIsInstance(result["optimizer"], FSDPOptimizerWrapper)
        self.assertIsInstance(result["optim2"], torch.optim.Optimizer)
        self.assertIsInstance(result["lr_scheduler"], TLRScheduler)

        with patch(
            "torchtnt.framework.unit._is_fsdp_module", side_effect=lambda m: m == module
        ), patch(
            "torchtnt.framework.unit._is_fsdp2_module",
            side_effect=lambda m: m == module,
        ):
            result = auto_unit._construct_tracked_optimizers_and_schedulers()

        self.assertIsInstance(result["optimizer"], FSDP2OptimizerWrapper)
        self.assertIsInstance(result["optim2"], torch.optim.Optimizer)
        self.assertIsInstance(result["lr_scheduler"], TLRScheduler)
