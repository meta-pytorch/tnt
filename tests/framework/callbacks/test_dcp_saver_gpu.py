#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import shutil
import tempfile
import unittest
from typing import Tuple
from unittest.mock import MagicMock, patch

import torch
from torch import distributed as dist, nn
from torchtnt.framework._test_utils import DummyAutoUnit, generate_random_dataloader
from torchtnt.framework.callbacks.dcp_saver import DistributedCheckpointSaver
from torchtnt.framework.train import train
from torchtnt.utils.distributed import get_global_rank, spawn_multi_process
from torchtnt.utils.lr_scheduler import TLRScheduler
from torchtnt.utils.prepare_module import FSDPStrategy
from torchtnt.utils.test_utils import skip_if_not_distributed, skip_if_not_gpu


class DistributedCheckpointSaverGPUTest(unittest.TestCase):
    @skip_if_not_distributed
    @skip_if_not_gpu
    def test_test_gloo_pg_restore(self) -> None:
        spawn_multi_process(
            1,
            "nccl",
            self._test_gloo_pg_restore,
        )

    @staticmethod
    @patch("torch.distributed.destroy_process_group")
    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def _test_gloo_pg_restore(
        mock_dist_cp: MagicMock, mock_destroy_process_group: MagicMock
    ) -> None:
        tc = unittest.TestCase()
        my_unit = DummyAutoUnit(module=nn.Linear(2, 3))
        DistributedCheckpointSaver.restore(path="path/to/snapshot", unit=my_unit)
        process_group = mock_dist_cp.load.call_args.kwargs["process_group"]
        tc.assertEqual(dist.get_backend(process_group), dist.Backend.GLOO, None)
        mock_destroy_process_group.assert_called_once()

    @skip_if_not_distributed
    @skip_if_not_gpu
    def test_test_gloo_pg_restore_wth_id(self) -> None:
        spawn_multi_process(
            1,
            "nccl",
            self._test_gloo_pg_restore,
        )

    @staticmethod
    @patch("torch.distributed.destroy_process_group")
    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def _test_gloo_pg_restore_with_id(
        mock_dist_cp: MagicMock, mock_destroy_process_group: MagicMock
    ) -> None:
        tc = unittest.TestCase()
        my_unit = DummyAutoUnit(module=nn.Linear(2, 3))
        DistributedCheckpointSaver.restore_with_id(
            checkpoint_id="path/to/snapshot", unit=my_unit
        )
        process_group = mock_dist_cp.load.call_args.kwargs["process_group"]
        tc.assertEqual(dist.get_backend(process_group), dist.Backend.GLOO, None)
        mock_destroy_process_group.assert_called_once()

    @skip_if_not_distributed
    @skip_if_not_gpu
    def test_save_restore_fsdp(self) -> None:
        spawn_multi_process(
            2,
            "nccl",
            self._save_restore_fsdp,
        )

    @staticmethod
    def _save_restore_fsdp() -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2
        save_every_n_epochs = 1

        my_unit = DummyAutoUnit(module=torch.nn.Linear(input_dim, 2), strategy="fsdp")
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        if get_global_rank() == 0:
            temp_dir = tempfile.mkdtemp()
        else:
            temp_dir = ""

        dcp_cb = DistributedCheckpointSaver(
            temp_dir,
            save_every_n_epochs=save_every_n_epochs,
        )
        temp_dir = dcp_cb.dirpath
        train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[dcp_cb])

        tc = unittest.TestCase()
        try:
            my_new_unit = DummyAutoUnit(
                module=torch.nn.Linear(input_dim, 2), strategy="fsdp"
            )
            tc.assertNotEqual(
                # pyrefly: ignore [missing-attribute]
                my_new_unit.optimizer.state_dict(),
                # pyrefly: ignore [missing-attribute]
                my_unit.optimizer.state_dict(),
            )
            # get latest checkpoint
            ckpt_path = os.path.join(temp_dir, f"epoch_{max_epochs}_train_step_10")
            dcp_cb.restore(ckpt_path, my_new_unit)
            tc.assertEqual(
                # pyrefly: ignore [missing-attribute]
                my_new_unit.optimizer.state_dict(),
                # pyrefly: ignore [missing-attribute]
                my_unit.optimizer.state_dict(),
            )
        finally:
            dist.barrier()  # avoid race condition
            if get_global_rank() == 0:
                shutil.rmtree(temp_dir)  # delete temp directory

    @skip_if_not_distributed
    @skip_if_not_gpu
    def test_save_restore_fsdp_orig_params_multiple_param_groups(self) -> None:
        spawn_multi_process(
            2,
            "nccl",
            self._save_restore_fsdp_orig_params_multiple_param_groups,
        )

    @staticmethod
    def _save_restore_fsdp_orig_params_multiple_param_groups() -> None:
        input_dim = 16

        class TwoParamGroupAutoUnit(DummyAutoUnit):
            def configure_optimizers_and_lr_scheduler(
                self, module: nn.Module
            ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
                params = list(module.parameters())
                optimizer = torch.optim.AdamW(  # noqa: CITRINE(missing_for_each_optimizer)
                    [
                        {"params": params[:3]},
                        {"params": params[3:], "weight_decay": 0.0},
                    ],
                    lr=0.01,
                )
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.9
                )
                return optimizer, lr_scheduler

        def make_unit() -> TwoParamGroupAutoUnit:
            # The no-decay norm is last in the flat param, so its data is only in
            # rank 1's shard; ranks disagreed on the optimizer match by data_ptr().
            return TwoParamGroupAutoUnit(
                module=nn.Sequential(
                    nn.Linear(input_dim, input_dim, bias=False),
                    nn.Linear(input_dim, input_dim, bias=False),
                    nn.Linear(input_dim, input_dim, bias=False),
                    nn.LayerNorm(input_dim, bias=False),
                ),
                strategy=FSDPStrategy(use_orig_params=True),
            )

        my_unit = make_unit()
        dataloader = generate_random_dataloader(10, input_dim, 2)
        temp_dir = tempfile.mkdtemp() if get_global_rank() == 0 else ""
        dcp_cb = DistributedCheckpointSaver(temp_dir, save_every_n_epochs=1)
        temp_dir = dcp_cb.dirpath
        train(my_unit, dataloader, max_epochs=1, callbacks=[dcp_cb])

        tc = unittest.TestCase()
        try:
            my_new_unit = make_unit()
            tc.assertTrue(
                DistributedCheckpointSaver.restore_from_latest(temp_dir, my_new_unit)
            )
            torch.testing.assert_close(
                # pyrefly: ignore [missing-attribute]
                my_new_unit.optimizer.state_dict(),
                # pyrefly: ignore [missing-attribute]
                my_unit.optimizer.state_dict(),
            )
        finally:
            dist.barrier()  # avoid race condition
            if get_global_rank() == 0:
                shutil.rmtree(temp_dir)  # delete temp directory

    @skip_if_not_distributed
    @skip_if_not_gpu
    def test_save_restore_fsdp_with_id(self) -> None:
        spawn_multi_process(
            2,
            "nccl",
            self._save_restore_fsdp_with_id,
        )

    @staticmethod
    def _save_restore_fsdp_with_id() -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2
        save_every_n_epochs = 1

        my_unit = DummyAutoUnit(module=torch.nn.Linear(input_dim, 2), strategy="fsdp")
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        if get_global_rank() == 0:
            temp_dir = tempfile.mkdtemp()
        else:
            temp_dir = ""

        dcp_cb = DistributedCheckpointSaver(
            temp_dir,
            save_every_n_epochs=save_every_n_epochs,
        )
        temp_dir = dcp_cb.dirpath
        train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[dcp_cb])

        tc = unittest.TestCase()
        try:
            my_new_unit = DummyAutoUnit(
                module=torch.nn.Linear(input_dim, 2), strategy="fsdp"
            )
            tc.assertNotEqual(
                # pyrefly: ignore [missing-attribute]
                my_new_unit.optimizer.state_dict(),
                # pyrefly: ignore [missing-attribute]
                my_unit.optimizer.state_dict(),
            )
            # get latest checkpoint
            ckpt_path = os.path.join(temp_dir, f"epoch_{max_epochs}_train_step_10")
            dcp_cb.restore_with_id(checkpoint_id=ckpt_path, unit=my_new_unit)
            tc.assertEqual(
                # pyrefly: ignore [missing-attribute]
                my_new_unit.optimizer.state_dict(),
                # pyrefly: ignore [missing-attribute]
                my_unit.optimizer.state_dict(),
            )
        finally:
            dist.barrier()  # avoid race condition
            if get_global_rank() == 0:
                shutil.rmtree(temp_dir)  # delete temp directory
