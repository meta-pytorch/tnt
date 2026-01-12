#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, Dict

import torch
from torch import distributed as dist, nn
from torch.ao.quantization.pt2e.export_utils import model_is_exported
from torch.distributed import launcher
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchtnt.framework._loop_utils import (
    _is_done,
    _is_epoch_done,
    _maybe_set_distributed_sampler_epoch,
    _reason_epoch_completed,
    _reset_module_training_mode,
    _set_module_training_mode,
)
from torchtnt.framework._test_utils import generate_random_dataset
from torchtnt.utils.progress import Progress
from torchtnt.utils.test_utils import get_pet_launch_config


class LoopUtilsTest(unittest.TestCase):
    def test_maybe_set_distributed_sampler_epoch(self) -> None:
        config = get_pet_launch_config(2)
        launcher.elastic_launch(
            config, entrypoint=self._test_maybe_set_distributed_sampler_epoch
        )()

    @staticmethod
    def _test_maybe_set_distributed_sampler_epoch() -> None:
        """
        Test _maybe_set_distributed_sampler_epoch util function
        """
        dist.init_process_group("gloo")

        random_dataset = generate_random_dataset(10, 3)
        dummy_dataloader_with_distributed_sampler = DataLoader(
            random_dataset, sampler=DistributedSampler(random_dataset)
        )

        _maybe_set_distributed_sampler_epoch(
            dummy_dataloader_with_distributed_sampler, 20
        )

        sampler = cast(
            DistributedSampler[object],
            dummy_dataloader_with_distributed_sampler.sampler,
        )
        assert sampler.epoch == 20

    def test_set_module_training_mode(self) -> None:
        """
        Test _set_module_training_mode
        """
        module = nn.Linear(1, 1)
        loss_fn = nn.CrossEntropyLoss()

        tracked_modules: Dict[str, torch.nn.Module] = {
            "module": module,
            "loss_fn": loss_fn,
        }

        # set module training mode to False
        prior_module_train_states = _set_module_training_mode(tracked_modules, False)

        self.assertFalse(module.training)
        self.assertFalse(loss_fn.training)

        self.assertTrue(prior_module_train_states["module"])
        self.assertTrue(prior_module_train_states["loss_fn"])

        # set back to True
        prior_module_train_states = _set_module_training_mode(tracked_modules, True)

        self.assertTrue(module.training)
        self.assertTrue(loss_fn.training)

        self.assertFalse(prior_module_train_states["module"])
        self.assertFalse(prior_module_train_states["loss_fn"])

    def test_set_module_training_mode_qat(self) -> None:
        """
        Test _set_module_training_mode
        """

        # define a floating point model
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 4)

            def forward(self, x):
                x = self.fc(x)
                return x

        loss_fn = nn.CrossEntropyLoss()
        module = torch.export.export(M(), (torch.rand(4, 4),), strict=True).module()

        tracked_modules: Dict[str, torch.nn.Module] = {
            "module": module,
            "loss_fn": loss_fn,
        }

        self.assertTrue(model_is_exported(module))
        prior_module_train_states = _set_module_training_mode(tracked_modules, False)

        self.assertFalse(module.training)
        self.assertFalse(loss_fn.training)

        self.assertTrue(prior_module_train_states["module"])
        self.assertTrue(prior_module_train_states["loss_fn"])

        # set back to True
        prior_module_train_states = _set_module_training_mode(tracked_modules, True)

        self.assertTrue(module.training)
        self.assertTrue(loss_fn.training)

        self.assertFalse(prior_module_train_states["module"])
        self.assertFalse(prior_module_train_states["loss_fn"])

    def test_reset_module_training_mode(self) -> None:
        """
        Test _reset_module_training_mode
        """
        module = nn.Linear(1, 1)
        loss_fn = nn.CrossEntropyLoss()

        tracked_modules: Dict[str, torch.nn.Module] = {
            "module": module,
            "loss_fn": loss_fn,
        }

        # set module training mode to False
        prior_module_train_states = _set_module_training_mode(tracked_modules, False)

        self.assertFalse(module.training)
        self.assertFalse(loss_fn.training)

        # set back to True using reset
        _reset_module_training_mode(tracked_modules, prior_module_train_states)

        self.assertTrue(module.training)
        self.assertTrue(loss_fn.training)

    def test_is_done(self) -> None:
        p = Progress(
            num_epochs_completed=2,
            num_steps_completed=100,
            num_steps_completed_in_epoch=5,
        )

        self.assertTrue(_is_done(p, max_epochs=2, max_steps=200))
        self.assertTrue(_is_done(p, max_epochs=2, max_steps=None))
        self.assertTrue(_is_done(p, max_epochs=3, max_steps=100))
        self.assertTrue(_is_done(p, max_epochs=None, max_steps=100))

        self.assertFalse(_is_done(p, max_epochs=3, max_steps=200))
        self.assertFalse(_is_done(p, max_epochs=None, max_steps=200))
        self.assertFalse(_is_done(p, max_epochs=3, max_steps=None))
        self.assertFalse(_is_done(p, max_epochs=None, max_steps=None))

    def test_is_epoch_done(self) -> None:
        p = Progress(
            num_epochs_completed=2,
            num_steps_completed=100,
            num_steps_completed_in_epoch=5,
        )

        self.assertTrue(_is_epoch_done(p, max_steps_per_epoch=5, max_steps=200))
        self.assertTrue(_is_epoch_done(p, max_steps_per_epoch=5, max_steps=None))
        self.assertTrue(_is_epoch_done(p, max_steps_per_epoch=100, max_steps=100))
        self.assertTrue(_is_epoch_done(p, max_steps_per_epoch=None, max_steps=100))

        self.assertFalse(_is_epoch_done(p, max_steps_per_epoch=6, max_steps=200))
        self.assertFalse(_is_epoch_done(p, max_steps_per_epoch=None, max_steps=200))
        self.assertFalse(_is_epoch_done(p, max_steps_per_epoch=6, max_steps=None))
        self.assertFalse(_is_epoch_done(p, max_steps_per_epoch=None, max_steps=None))

    def test_log_reason_epoch_completed(self) -> None:
        p = Progress(
            num_epochs_completed=2,
            num_steps_completed=100,
            num_steps_completed_in_epoch=5,
        )

        reason = _reason_epoch_completed(
            p, max_steps_per_epoch=5, max_steps=None, stop_iteration_reached=False
        )
        self.assertEqual(
            reason, "Train epoch 3 ended as max steps per epoch reached: 5"
        )

        reason = _reason_epoch_completed(
            p, max_steps_per_epoch=6, max_steps=100, stop_iteration_reached=False
        )
        self.assertEqual(reason, "Train epoch 3 ended as max steps reached: 100")

        reason = _reason_epoch_completed(
            p, max_steps_per_epoch=5, max_steps=None, stop_iteration_reached=True
        )
        self.assertEqual(
            reason, "Train epoch 3 ended as it reached end of train dataloader"
        )
