#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from copy import deepcopy
from typing import Tuple, TypeVar
from unittest.mock import MagicMock, patch

import torch
from pyre_extensions import ParameterSpecification as ParamSpec
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torchtnt.framework._test_utils import (
    DummyAutoUnit,
    generate_random_dataloader,
    generate_random_iterable_dataloader,
    get_dummy_train_state,
)
from torchtnt.framework.auto_unit import AutoPredictUnit, SWALRParams, SWAParams
from torchtnt.framework.evaluate import evaluate
from torchtnt.framework.fit import fit
from torchtnt.framework.predict import predict
from torchtnt.framework.state import ActivePhase, State
from torchtnt.framework.train import train
from torchtnt.utils.distributed import spawn_multi_process
from torchtnt.utils.env import init_from_env, seed
from torchtnt.utils.prepare_module import DDPStrategy, FSDPStrategy, TorchCompileParams
from torchtnt.utils.test_utils import skip_if_not_distributed, skip_if_not_gpu

TParams = ParamSpec("TParams")
T = TypeVar("T")


Batch = Tuple[torch.Tensor, torch.Tensor]


class DummySWAAutoUnit(DummyAutoUnit):
    def compute_loss(self, state: State, data: Batch) -> Tuple[torch.Tensor, object]:
        """
        Computes loss for given batch. If in EVAL or PREDICT phase, uses swa model's output
        """
        inputs, targets = data
        if state.active_phase == ActivePhase.TRAIN:
            outputs = self.module(inputs)
        else:
            outputs = self.swa_model(inputs) if self.swa_model else self.module(inputs)

        loss = torch.nn.functional.cross_entropy(outputs, targets)

        return loss, outputs


class TestAutoUnitGPU(unittest.TestCase):
    @skip_if_not_gpu
    @skip_if_not_distributed
    def test_fsdp_fp16(self) -> None:
        """
        Test that FSDP + FP16 uses ShardedGradScaler
        """
        spawn_multi_process(
            2,
            "nccl",
            self._test_fsdp_fp16,
        )

    @staticmethod
    def _test_fsdp_fp16() -> None:
        device = init_from_env()
        my_module = torch.nn.Linear(2, 2)
        auto_unit_fsdp = DummyAutoUnit(
            module=my_module,
            device=device,
            strategy=FSDPStrategy(),
            precision="fp16",
        )
        tc = unittest.TestCase()
        tc.assertTrue(isinstance(auto_unit_fsdp.grad_scaler, ShardedGradScaler))

    @skip_if_not_gpu
    @patch("torch.autocast")
    def test_mixed_precision_fp16(self, mock_autocast: MagicMock) -> None:
        """
        Test that the mixed precision autocast context is called when fp16 precision is set
        """
        my_module = torch.nn.Linear(2, 2)
        auto_unit = DummyAutoUnit(
            module=my_module,
            precision="fp16",
        )
        dummy_iterable = [(torch.ones(2, 2), torch.ones(2, 2))]
        state = get_dummy_train_state(dummy_iterable)
        auto_unit.train_step(
            state=state,
            data=auto_unit.get_next_train_batch(state, iter(dummy_iterable)),
        )
        mock_autocast.assert_called_with(
            device_type="cuda", dtype=torch.float16, enabled=True
        )

    @skip_if_not_gpu
    @patch("torch.autocast")
    def test_mixed_precision_bf16(self, mock_autocast: MagicMock) -> None:
        """
        Test that the mixed precision autocast context is called when bf16 precision is set
        """
        my_module = torch.nn.Linear(2, 2)

        auto_unit = DummyAutoUnit(
            module=my_module,
            precision="bf16",
        )
        dummy_iterable = [(torch.ones(2, 2), torch.ones(2, 2))]
        state = get_dummy_train_state(dummy_iterable)
        auto_unit.train_step(
            state=state,
            data=auto_unit.get_next_train_batch(state, iter(dummy_iterable)),
        )
        mock_autocast.assert_called_with(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        )

    @skip_if_not_distributed
    @skip_if_not_gpu
    def test_stochastic_weight_averaging_fsdp(self) -> None:
        """
        Test that swa params with FSDP is identical to non-FSDP swa
        """
        spawn_multi_process(
            2,
            "nccl",
            self._test_stochastic_weight_averaging_fsdp,
        )

    @staticmethod
    def _test_stochastic_weight_averaging_fsdp() -> None:
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l1 = torch.nn.Linear(2, 2)
                self.b1 = torch.nn.BatchNorm1d(2)
                self.l2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                x = self.l1(x)
                x = self.b1(x)
                x = self.l2(x)
                return x

        # so all ranks start with same initialized weights
        device = init_from_env()
        seed(0)
        my_module = Net()

        auto_unit = DummyAutoUnit(
            module=deepcopy(my_module),
            device=device,
            step_lr_interval="step",
            swa_params=SWAParams(
                warmup_steps_or_epochs=1,
                step_or_epoch_update_freq=1,
                swalr_params=SWALRParams(
                    anneal_steps_or_epochs=3,
                ),
                averaging_method="ema",
            ),
        )

        auto_unit_fsdp = DummyAutoUnit(
            module=my_module,
            device=device,
            step_lr_interval="step",
            strategy=FSDPStrategy(),
            swa_params=SWAParams(
                warmup_steps_or_epochs=1,
                step_or_epoch_update_freq=1,
                swalr_params=SWALRParams(
                    anneal_steps_or_epochs=3,
                ),
                averaging_method="ema",
            ),
        )

        input_dim = 2
        dataset_len = 10
        batch_size = 2

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        train(auto_unit, dataloader, max_epochs=1, max_steps_per_epoch=5)
        train(auto_unit_fsdp, dataloader, max_epochs=1, max_steps_per_epoch=5)

        swa_params = list(auto_unit.swa_model.module.parameters())
        with FSDP.summon_full_params(auto_unit_fsdp.swa_model):
            swa_fsdp_params = list(auto_unit_fsdp.swa_model.module.parameters())

            # Iterate and compare each parameter
            for p1, p2 in zip(swa_params, swa_fsdp_params, strict=True):
                torch.testing.assert_close(p2, p1, check_device=False)

    @skip_if_not_distributed
    @skip_if_not_gpu
    def test_stochastic_weight_averaging_fsdp_with_eval(self) -> None:
        """
        Test that swa params with FSDP is identical to non-FSDP swa
        """
        spawn_multi_process(
            2,
            "nccl",
            self._test_stochastic_weight_averaging_fsdp_with_eval,
        )

    @staticmethod
    def _test_stochastic_weight_averaging_fsdp_with_eval() -> None:
        """
        Compares the swa model parameters after training without FSDP and with FSDP.
        They should be identical.
        """

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l1 = torch.nn.Linear(2, 2)
                self.b1 = torch.nn.BatchNorm1d(2)
                self.l2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                x = self.l1(x)
                x = self.b1(x)
                x = self.l2(x)
                return x

        # so all ranks start with same initialized weights
        device = init_from_env()
        seed(0)
        my_module = Net()

        auto_unit = DummySWAAutoUnit(
            module=deepcopy(my_module),
            device=device,
            step_lr_interval="step",
            swa_params=SWAParams(
                warmup_steps_or_epochs=1,
                step_or_epoch_update_freq=1,
                swalr_params=SWALRParams(
                    anneal_steps_or_epochs=3,
                ),
                averaging_method="ema",
            ),
        )

        auto_unit_fsdp = DummySWAAutoUnit(
            module=my_module,
            device=device,
            step_lr_interval="step",
            strategy=FSDPStrategy(),
            swa_params=SWAParams(
                warmup_steps_or_epochs=1,
                step_or_epoch_update_freq=1,
                swalr_params=SWALRParams(
                    anneal_steps_or_epochs=3,
                ),
                averaging_method="ema",
            ),
        )

        input_dim = 2
        dataset_len = 10
        batch_size = 2

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        eval_dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        fit(
            auto_unit,
            dataloader,
            eval_dataloader,
            max_epochs=3,
            max_train_steps_per_epoch=5,
            evaluate_every_n_epochs=0,
        )

        fit(
            auto_unit_fsdp,
            dataloader,
            eval_dataloader,
            max_epochs=3,
            max_train_steps_per_epoch=5,
            # this is key arg, to ensure that swa model is updated
            # even after swa model forward pass is used in eval
            evaluate_every_n_epochs=1,
        )

        swa_params = list(auto_unit.swa_model.parameters())
        swa_buffers = list(auto_unit.swa_model.buffers())
        with FSDP.summon_full_params(auto_unit_fsdp.swa_model):
            swa_fsdp_params = auto_unit_fsdp.swa_model.parameters()
            swa_fsdp_buffers = auto_unit_fsdp.swa_model.buffers()

            # Iterate and compare each parameter
            for p1, p2 in zip(swa_params, swa_fsdp_params, strict=True):
                torch.testing.assert_close(p2, p1, check_device=False)

            # Iterate and compare each buffer
            for b1, b2 in zip(swa_buffers, swa_fsdp_buffers, strict=True):
                torch.testing.assert_close(b2, b1, check_device=False)

    @skip_if_not_gpu
    @patch("torch.autocast")
    def test_eval_mixed_precision_bf16(self, mock_autocast: MagicMock) -> None:
        """
        Test that the mixed precision autocast context is called during evaluate when precision = bf16
        """
        my_module = torch.nn.Linear(2, 2)
        auto_unit = DummyAutoUnit(
            module=my_module,
            precision="bf16",
        )

        input_dim = 2
        dataset_len = 8
        batch_size = 2

        eval_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)
        evaluate(auto_unit, eval_dl)
        mock_autocast.assert_called_with(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        )

    @skip_if_not_gpu
    @skip_if_not_distributed
    def test_no_sync(self) -> None:
        """
        Test that the no_sync autocast context is correctly applied when using gradient accumulation
        """
        spawn_multi_process(
            2,
            "nccl",
            self._test_ddp_no_sync,
        )
        spawn_multi_process(
            2,
            "nccl",
            self._test_fsdp_no_sync,
        )

    @staticmethod
    def _test_ddp_no_sync() -> None:
        """
        Test that the no_sync autocast context is correctly applied when using gradient accumulation and DDP
        """

        my_module = torch.nn.Linear(2, 2)

        auto_unit = DummyAutoUnit(
            module=my_module,
            strategy=DDPStrategy(),
            gradient_accumulation_steps=2,
        )

        dummy_iterator = iter(
            [(torch.ones(2, 2), torch.ones(2, 2)), (torch.ones(2, 2), torch.ones(2, 2))]
        )
        state = get_dummy_train_state()

        # for the first step no_sync should be called since we accumulate gradients
        with patch.object(auto_unit.module, "no_sync") as no_sync_mock:
            auto_unit.train_step(
                state=state, data=auto_unit.get_next_train_batch(state, dummy_iterator)
            )
            no_sync_mock.assert_called_once()

        auto_unit.train_progress.increment_step()
        # for the second step no_sync should not be called since we run optimizer step
        with patch.object(auto_unit.module, "no_sync") as no_sync_mock:
            auto_unit.train_step(
                state=state, data=auto_unit.get_next_train_batch(state, dummy_iterator)
            )
            no_sync_mock.assert_not_called()

    @staticmethod
    def _test_fsdp_no_sync() -> None:
        """
        Test that the no_sync autocast context is correctly applied when using gradient accumulation and FSDP
        """
        device = init_from_env()
        my_module = torch.nn.Linear(2, 2).to(device)

        auto_unit = DummyAutoUnit(
            module=my_module,
            device=device,
            strategy=FSDPStrategy(),
            gradient_accumulation_steps=2,
        )

        dummy_iterator = iter(
            [(torch.ones(2, 2), torch.ones(2, 2)), (torch.ones(2, 2), torch.ones(2, 2))]
        )
        state = get_dummy_train_state()

        # for the first step no_sync should be called since we accumulate gradients
        with patch.object(auto_unit.module, "no_sync") as no_sync_mock:
            auto_unit.train_step(
                state=state, data=auto_unit.get_next_train_batch(state, dummy_iterator)
            )
            no_sync_mock.assert_called_once()

        auto_unit.train_progress.increment_step()
        # for the second step no_sync should not be called since we run optimizer step
        with patch.object(auto_unit.module, "no_sync") as no_sync_mock:
            auto_unit.train_step(
                state=state, data=auto_unit.get_next_train_batch(state, dummy_iterator)
            )
            no_sync_mock.assert_not_called()

    @skip_if_not_gpu
    @patch("torch.autocast")
    def test_predict_mixed_precision_fp16(self, mock_autocast: MagicMock) -> None:
        """
        Test that the mixed precision autocast context is called during predict when precision = fp16
        """
        my_module = torch.nn.Linear(2, 2)
        auto_unit = AutoPredictUnit(module=my_module, precision="fp16")

        input_dim = 2
        dataset_len = 8
        batch_size = 2

        predict_dl = generate_random_iterable_dataloader(
            dataset_len, input_dim, batch_size
        )
        predict(auto_unit, predict_dl)
        mock_autocast.assert_called_with(
            device_type="cuda", dtype=torch.float16, enabled=True
        )

    @skip_if_not_gpu
    @patch("torch.compile")
    def test_compile_predict(self, mock_dynamo: MagicMock) -> None:
        """
        e2e torch compile on predict
        """
        my_module = torch.nn.Linear(2, 2)
        auto_unit = AutoPredictUnit(
            module=my_module,
            torch_compile_params=TorchCompileParams(backend="eager"),
        )

        input_dim = 2
        dataset_len = 8
        batch_size = 2

        predict_dl = generate_random_iterable_dataloader(
            dataset_len, input_dim, batch_size
        )
        predict(auto_unit, predict_dl)
        mock_dynamo.assert_called()
