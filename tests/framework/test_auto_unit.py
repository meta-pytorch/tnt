#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Literal, Optional, Tuple, TypeVar
from unittest.mock import MagicMock, patch

import torch

from pyre_extensions import none_throws, ParameterSpecification as ParamSpec

from torch.distributed import GradBucket
from torchtnt.framework._test_utils import (
    DummyAutoUnit,
    generate_random_dataloader,
    generate_random_iterable_dataloader,
    get_dummy_train_state,
)

from torchtnt.framework.auto_unit import (
    AutoPredictUnit,
    AutoUnit,
    SWALRParams,
    SWAParams,
    TrainStepResults,
)
from torchtnt.framework.evaluate import evaluate
from torchtnt.framework.predict import predict
from torchtnt.framework.state import ActivePhase, State
from torchtnt.framework.train import train
from torchtnt.framework.unit import TPredictData
from torchtnt.utils.device import copy_data_to_device
from torchtnt.utils.distributed import spawn_multi_process
from torchtnt.utils.env import init_from_env
from torchtnt.utils.lr_scheduler import TLRScheduler
from torchtnt.utils.prepare_module import DDPStrategy
from torchtnt.utils.progress import Progress
from torchtnt.utils.test_utils import skip_if_not_distributed
from torchtnt.utils.timer import Timer

TParams = ParamSpec("TParams")
T = TypeVar("T")


class TestAutoUnit(unittest.TestCase):
    def test_app_state_mixin(self) -> None:
        """
        Test that app_state, tracked_optimizers, tracked_lr_schedulers are set as expected with AutoUnit
        """
        my_module = torch.nn.Linear(2, 2)

        auto_unit = DummyAutoUnit(
            module=my_module,
            precision="fp16",
        )

        self.assertEqual(auto_unit.tracked_modules()["module"], my_module)
        self.assertTrue(
            isinstance(
                auto_unit.tracked_misc_statefuls()["grad_scaler"],
                torch.cuda.amp.GradScaler,
            )
        )
        for key in ("module", "optimizer", "lr_scheduler", "grad_scaler"):
            self.assertIn(key, auto_unit.app_state())

    def test_lr_scheduler_step(self) -> None:
        """
        Test that the lr scheduler is stepped every optimizer step when step_lr_interval="step"
        """
        my_module = torch.nn.Linear(2, 2)

        auto_unit = DummyLRSchedulerAutoUnit(
            module=my_module,
            step_lr_interval="step",
        )

        input_dim = 2
        dataset_len = 8
        batch_size = 2
        max_epochs = 3
        expected_steps_per_epoch = dataset_len / batch_size * max_epochs

        train_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)
        train(auto_unit, train_dataloader=train_dl, max_epochs=max_epochs)
        self.assertEqual(
            auto_unit.lr_scheduler.step.call_count, expected_steps_per_epoch
        )

    def test_lr_scheduler_epoch(self) -> None:
        """
        Test that the lr scheduler is stepped every epoch when step_lr_interval="epoch"
        """
        my_module = torch.nn.Linear(2, 2)

        auto_unit = DummyLRSchedulerAutoUnit(
            module=my_module,
            step_lr_interval="epoch",
        )

        input_dim = 2
        dataset_len = 8
        batch_size = 2
        max_epochs = 3

        train_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)

        train(auto_unit, train_dataloader=train_dl, max_epochs=max_epochs)
        self.assertEqual(auto_unit.lr_scheduler.step.call_count, max_epochs)

    def test_mixed_precision_invalid_str(self) -> None:
        """
        Test that an exception is raised with an invalid precision string
        """
        my_module = torch.nn.Linear(2, 2)

        with self.assertRaisesRegex(ValueError, "Precision foo not supported"):
            _ = DummyAutoUnit(
                module=my_module,
                precision="foo",
            )

    def test_predict_step(self) -> None:
        """
        Test predict step functionality
        """
        my_module = torch.nn.Linear(2, 2)
        auto_unit = DummyAutoUnit(
            module=my_module,
        )

        input_dim = 2
        dataset_len = 10
        batch_size = 2

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        dataloader_iter = iter(dataloader)
        pred_dataloader = (x[0] for x in dataloader_iter)  # only need data, not target

        with patch(
            "torchtnt.framework._test_utils.DummyAutoUnit.on_predict_step_end"
        ) as mock_predict_step_end:
            predict(auto_unit, pred_dataloader, max_steps_per_epoch=1)
            mock_predict_step_end.assert_called_once()

    def test_stochastic_weight_averaging_basic(self) -> None:
        """
        Basic stochastic weight averaging tests
        """
        my_module = torch.nn.Linear(2, 2)

        auto_unit = DummyAutoUnit(
            module=my_module,
        )

        auto_unit2 = DummyAutoUnit(
            module=my_module,
            swa_params=SWAParams(
                warmup_steps_or_epochs=1,
                step_or_epoch_update_freq=1,
                swalr_params=SWALRParams(
                    anneal_steps_or_epochs=5,
                ),
            ),
        )

        self.assertIsNone(auto_unit.swa_model)
        self.assertIsNotNone(auto_unit2.swa_model)

        self.assertNotIn("swa_model", auto_unit.app_state())
        self.assertNotIn("swa_model", auto_unit.tracked_modules())
        self.assertNotIn("swa_scheduler", auto_unit.app_state())
        self.assertNotIn("swa_scheduler", auto_unit.tracked_lr_schedulers())
        self.assertIn("swa_model", auto_unit2.app_state())
        self.assertIn("swa_model", auto_unit2.tracked_modules())
        self.assertIn("swa_scheduler", auto_unit2.app_state())
        self.assertIn("swa_scheduler", auto_unit2.tracked_lr_schedulers())

    def test_stochastic_weight_averaging_update_freq(self) -> None:
        """
        e2e stochastic weight averaging test to ensure that the SWA model is updated at the correct frequency
        """

        my_module = torch.nn.Linear(2, 2)
        swa_params = SWAParams(
            warmup_steps_or_epochs=2,
            step_or_epoch_update_freq=1,
            swalr_params=SWALRParams(
                anneal_steps_or_epochs=5,
            ),
            averaging_method="ema",
        )
        auto_unit = DummyAutoUnit(
            module=my_module,
            step_lr_interval="step",
            swa_params=swa_params,
        )

        input_dim = 2
        dataset_len = 12
        batch_size = 1

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        with patch.object(auto_unit, "_update_swa") as update_swa_mock:
            train(auto_unit, dataloader, max_epochs=1, max_steps_per_epoch=4)
            # 2 warmup + 2 steps = 4
            self.assertEqual(update_swa_mock.call_count, 2)

        auto_unit.train_progress = Progress()  # reset progress
        auto_unit.swa_params.step_or_epoch_update_freq = 2
        with patch.object(auto_unit, "_update_swa") as update_swa_mock:
            train(auto_unit, dataloader, max_epochs=1, max_steps_per_epoch=6)
            # 2 warmup + step 4 + step 6 = 2
            self.assertEqual(update_swa_mock.call_count, 2)

        auto_unit.step_lr_interval = "epoch"
        auto_unit.train_progress = Progress()  # reset progress
        auto_unit.swa_params.warmup_steps_or_epochs = 1
        auto_unit.swa_params.step_or_epoch_update_freq = 1
        with patch.object(auto_unit, "_update_swa") as update_swa_mock:
            with patch.object(auto_unit.lr_scheduler, "step") as lr_scheduler_step_mock:
                train(auto_unit, dataloader, max_epochs=3, max_steps_per_epoch=2)
                self.assertEqual(lr_scheduler_step_mock.call_count, 1)
            # 1 warmup + epoch 2 + epoch 3 = 2
            self.assertEqual(update_swa_mock.call_count, 2)

    def test_move_data_to_device(self) -> None:
        """
        Test that move_data_to_device is called
        """
        device = init_from_env()
        my_module = torch.nn.Linear(2, 2)

        auto_unit = DummyAutoUnit(
            module=my_module,
            device=device,
        )

        state = get_dummy_train_state()

        dummy_data = (torch.ones(2, 2), torch.ones(2, 2))
        data_iter = iter([dummy_data])

        with patch.object(
            DummyAutoUnit, "move_data_to_device"
        ) as move_data_to_device_mock:
            dummy_data = copy_data_to_device(dummy_data, device)
            move_data_to_device_mock.return_value = dummy_data
            auto_unit._get_next_batch(state=state, data=data_iter)
            move_data_to_device_mock.assert_called_once()

    def test_configure_optimizers_and_lr_scheduler(self) -> None:
        """
        Test configure_optimizers_and_lr_scheduler with a complex AutoUnit where configure_optimizers_and_lr_scheduler uses an attribute set in the init
        """
        my_module = torch.nn.Linear(2, 2)

        auto_unit = DummyComplexAutoUnit(
            lr=0.01,
            module=my_module,
        )
        # assert that the optimizer attribute was correctly initialized and set
        self.assertTrue(hasattr(auto_unit, "optimizer"))
        self.assertTrue(hasattr(auto_unit, "lr_scheduler"))

    def test_configure_optimizers_and_lr_scheduler_called_once(self) -> None:
        """
        Test configure_optimizers_and_lr_scheduler is called exactly once
        """
        my_module = torch.nn.Linear(2, 2)

        with patch.object(
            DummyComplexAutoUnit, "configure_optimizers_and_lr_scheduler"
        ) as configure_optimizers_and_lr_scheduler_mock:
            configure_optimizers_and_lr_scheduler_mock.return_value = (
                MagicMock(),
                MagicMock(),
            )
            _ = DummyComplexAutoUnit(
                lr=0.01,
                module=my_module,
            )
            self.assertEqual(configure_optimizers_and_lr_scheduler_mock.call_count, 1)

    @skip_if_not_distributed
    def test_auto_unit_ddp(self) -> None:
        """
        Launch tests of AutoUnit with DDP strategy
        """

        spawn_multi_process(
            2,
            "gloo",
            self._test_stochastic_weight_averaging_with_ddp,
        )
        spawn_multi_process(
            2,
            "gloo",
            self._test_ddp_comm_hook,
        )

    @staticmethod
    def _test_stochastic_weight_averaging_with_ddp() -> None:
        """
        e2e stochastic weight averaging test with ddp
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

        my_module = Net()
        my_swa_params = SWAParams(
            warmup_steps_or_epochs=1,
            step_or_epoch_update_freq=1,
            swalr_params=SWALRParams(
                anneal_steps_or_epochs=3,
            ),
        )

        auto_unit = DummyAutoUnit(
            module=my_module,
            strategy=DDPStrategy(),
            swa_params=my_swa_params,
        )

        input_dim = 2
        dataset_len = 10
        batch_size = 2

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        train(auto_unit, dataloader, max_epochs=5, max_steps_per_epoch=1)

    @staticmethod
    def _test_ddp_comm_hook() -> None:
        """
        Test communication hook for DDP
        """
        custom_noop_hook_called = False

        def custom_noop_hook(
            _: Any, bucket: GradBucket
        ) -> torch.futures.Future[torch.Tensor]:
            nonlocal custom_noop_hook_called

            # pyre-fixme[29]: `Type[torch.futures.Future]` is not a function.
            fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
            fut.set_result(bucket.buffer())
            custom_noop_hook_called = True
            return fut

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

        my_module = Net()
        auto_unit = DummyAutoUnit(
            module=my_module,
            strategy=DDPStrategy(comm_hook=custom_noop_hook),
        )

        input_dim = 2
        dataset_len = 10
        batch_size = 2

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        train(auto_unit, dataloader, max_epochs=1, max_steps_per_epoch=1)

        tc = unittest.TestCase()
        tc.assertTrue(custom_noop_hook_called)

    def test_is_last_batch(self) -> None:
        """
        Test that is_last_batch is set correctly.
        """
        input_dim = 2
        dataset_len = 8
        batch_size = 2
        expected_steps_per_epoch = dataset_len / batch_size
        max_epochs = 1
        my_module = torch.nn.Linear(input_dim, 2)

        my_unit = LastBatchAutoUnit(
            module=my_module,
            expected_steps_per_epoch=expected_steps_per_epoch,
        )

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        train(my_unit, dataloader, max_epochs=max_epochs)

        my_unit = DummyAutoUnit(module=my_module)
        train(my_unit, dataloader, max_epochs=1, max_steps_per_epoch=4)
        self.assertFalse(my_unit._is_last_batch)

    def test_auto_unit_timing_train(self) -> None:
        """
        Test auto timing in AutoUnit for training
        """

        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 1
        max_epochs = 1

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        my_module = torch.nn.Linear(2, 2)

        train(
            TimingAutoUnit(module=my_module),
            dataloader,
            max_steps_per_epoch=max_steps_per_epoch,
            max_epochs=max_epochs,
            timer=Timer(),
        )

    def test_auto_unit_timing_eval(self) -> None:
        """
        Test auto timing in AutoUnit for eval
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 1

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        my_module = torch.nn.Linear(2, 2)

        evaluate(
            TimingAutoUnit(module=my_module),
            dataloader,
            max_steps_per_epoch=max_steps_per_epoch,
            timer=Timer(),
        )

    def test_auto_unit_timing_predict(self) -> None:
        """
        Test auto timing in AutoUnit for predict
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 1

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        dataloader_iter = iter(dataloader)
        pred_dataloader = (x[0] for x in dataloader_iter)  # only need data, not targets

        my_module = torch.nn.Linear(2, 2)

        predict(
            TimingAutoUnit(module=my_module),
            pred_dataloader,
            max_steps_per_epoch=max_steps_per_epoch,
            timer=Timer(),
        )

    def test_auto_predict_unit_timing_predict(self) -> None:
        """
        Test auto timing in AutoUnit for predict
        """
        my_module = torch.nn.Linear(2, 2)

        input_dim = 2
        dataset_len = 8
        batch_size = 2

        predict_dl = generate_random_iterable_dataloader(
            dataset_len, input_dim, batch_size
        )
        predict(
            TimingAutoPredictUnit(module=my_module),
            predict_dl,
            max_steps_per_epoch=1,
            timer=Timer(),
        )

    @patch("torch.autograd.set_detect_anomaly")
    def test_predict_detect_anomaly(self, mock_detect_anomaly: MagicMock) -> None:
        my_module = torch.nn.Linear(2, 2)
        auto_unit = AutoPredictUnit(module=my_module, detect_anomaly=True)

        input_dim = 2
        dataset_len = 8
        batch_size = 2

        predict_dl = generate_random_iterable_dataloader(
            dataset_len, input_dim, batch_size
        )
        predict(auto_unit, predict_dl, max_steps_per_epoch=1)
        mock_detect_anomaly.assert_called()

    def test_get_next_batch_with_single_phase(self) -> None:
        auto_unit = DummyAutoUnit(module=torch.nn.Linear(2, 2))
        first_data_iter = iter([1, 2])
        second_data_iter = iter([3])
        state = get_dummy_train_state()
        state._active_phase = ActivePhase.TRAIN
        self._assert_next_batch_dicts(auto_unit)

        move_data_to_device_mock = patch.object(
            auto_unit,
            "move_data_to_device",
            side_effect=lambda state, data, non_blocking: data,
        )

        with move_data_to_device_mock:
            batch = auto_unit._get_next_batch(state, first_data_iter)
        self.assertEqual(batch, 1)
        self._assert_next_batch_dicts(
            auto_unit, train_prefetched=True, train_next_batch=2
        )

        with move_data_to_device_mock:
            batch = auto_unit._get_next_batch(state, second_data_iter)
        # prefetched data is still from the previous data iter even though the new data iter is passed
        self.assertEqual(batch, 2)
        self._assert_next_batch_dicts(
            auto_unit, train_prefetched=True, train_next_batch=3
        )

        with move_data_to_device_mock:
            batch = auto_unit._get_next_batch(state, second_data_iter)
        self.assertEqual(batch, 3)
        self._assert_next_batch_dicts(auto_unit, train_prefetched=True)
        self.assertTrue(auto_unit._is_last_batch)

        with move_data_to_device_mock, self.assertRaises(StopIteration):
            auto_unit._get_next_batch(state, second_data_iter)
        self._assert_next_batch_dicts(auto_unit)
        self.assertFalse(auto_unit._is_last_batch)

    def test_get_next_batch_with_multiple_phases(self) -> None:
        auto_unit = DummyAutoUnit(module=torch.nn.Linear(2, 2))
        train_data_iter = iter([1, 2])
        eval_data_iter = iter([3, 4])
        predict_data_iter = iter([5, 6])
        state = get_dummy_train_state()
        state._active_phase = ActivePhase.TRAIN
        self._assert_next_batch_dicts(auto_unit)

        move_data_to_device_mock = patch.object(
            auto_unit,
            "move_data_to_device",
            side_effect=lambda state, data, non_blocking: data,
        )

        with move_data_to_device_mock:
            batch = auto_unit._get_next_batch(state, train_data_iter)
        self.assertEqual(batch, 1)
        self._assert_next_batch_dicts(
            auto_unit, train_prefetched=True, train_next_batch=2
        )

        state._active_phase = ActivePhase.EVALUATE
        with move_data_to_device_mock:
            batch = auto_unit._get_next_batch(state, eval_data_iter)
        self.assertEqual(batch, 3)
        self._assert_next_batch_dicts(
            auto_unit,
            train_prefetched=True,
            train_next_batch=2,
            eval_prefetched=True,
            eval_next_batch=4,
        )

        state._active_phase = ActivePhase.PREDICT
        with move_data_to_device_mock:
            batch = auto_unit._get_next_batch(state, predict_data_iter)
        self.assertEqual(batch, 5)
        self._assert_next_batch_dicts(
            auto_unit,
            train_prefetched=True,
            train_next_batch=2,
            eval_prefetched=True,
            eval_next_batch=4,
            predict_prefetched=True,
            predict_next_batch=6,
        )

    @staticmethod
    def _assert_next_batch_dicts(
        auto_unit: AutoUnit[T],
        *,
        train_prefetched: bool = False,
        eval_prefetched: bool = False,
        predict_prefetched: bool = False,
        train_next_batch: Optional[T] = None,
        eval_next_batch: Optional[T] = None,
        predict_next_batch: Optional[T] = None,
    ) -> None:
        tc = unittest.TestCase()
        tc.assertDictEqual(
            auto_unit._phase_to_prefetched,
            {
                ActivePhase.TRAIN: train_prefetched,
                ActivePhase.EVALUATE: eval_prefetched,
                ActivePhase.PREDICT: predict_prefetched,
            },
        )
        tc.assertDictEqual(
            auto_unit._phase_to_next_batch,
            {
                ActivePhase.TRAIN: train_next_batch,
                ActivePhase.EVALUATE: eval_next_batch,
                ActivePhase.PREDICT: predict_next_batch,
            },
        )


Batch = Tuple[torch.Tensor, torch.Tensor]


class DummyLRSchedulerAutoUnit(AutoUnit[Batch]):
    def __init__(
        self,
        module: torch.nn.Module,
        step_lr_interval: Literal["step", "epoch"] = "epoch",
    ) -> None:
        super().__init__(module=module, step_lr_interval=step_lr_interval)

    def compute_loss(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data
        outputs = self.module(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        return loss, outputs

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
        my_optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        my_lr_scheduler = MagicMock()
        return my_optimizer, my_lr_scheduler


class DummyComplexAutoUnit(AutoUnit[Batch]):
    def __init__(self, lr: float, module: torch.nn.Module) -> None:
        super().__init__(module=module)
        self.lr = lr

    def compute_loss(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data
        outputs = self.module(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        return loss, outputs

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
        my_optimizer = torch.optim.SGD(module.parameters(), lr=self.lr)
        my_lr_scheduler = MagicMock()
        return my_optimizer, my_lr_scheduler


class LastBatchAutoUnit(AutoUnit[Batch]):
    def __init__(self, module: torch.nn.Module, expected_steps_per_epoch: int) -> None:
        super().__init__(module=module)
        self.expected_steps_per_epoch = expected_steps_per_epoch
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def compute_loss(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tc = unittest.TestCase()
        tc.assertEqual(
            self._is_last_batch,
            self.train_progress.num_steps_completed_in_epoch + 1
            == self.expected_steps_per_epoch,
        )
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss, outputs

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
        my_optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        my_lr_scheduler = MagicMock()
        return my_optimizer, my_lr_scheduler


class TimingAutoUnit(AutoUnit[Batch]):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__(module=module)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def compute_loss(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss, outputs

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
        my_optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        my_lr_scheduler = MagicMock()
        return my_optimizer, my_lr_scheduler

    def on_train_step_end(
        self,
        state: State,
        data: Batch,
        step: int,
        results: TrainStepResults,
    ) -> None:
        assert state.train_state
        if self.train_progress.num_steps_completed_in_epoch == 1:
            tc = unittest.TestCase()
            recorded_timer_keys = none_throws(state.timer).recorded_durations.keys()
            for k in (
                "TimingAutoUnit.on_train_start",
                "TimingAutoUnit.on_train_epoch_start",
                "train.iter(dataloader)",
                "TimingAutoUnit.train.next(data_iter)",
                "TimingAutoUnit.train.move_data_to_device",
                "TimingAutoUnit.compute_loss",
                "TimingAutoUnit.backward",
                "TimingAutoUnit.optimizer_step",
                "TimingAutoUnit.optimizer_zero_grad",
                "TimingAutoUnit.on_train_step_end",
                "TimingAutoUnit.lr_scheduler_step",
            ):
                tc.assertIn(k, recorded_timer_keys)

            # train_step should not be in the timer's recorded_durations because it overlaps with other timings in the AutoUnit's train_step
            tc.assertNotIn("TimingAutoUnit.train_step", recorded_timer_keys)

    def on_eval_step_end(
        self,
        state: State,
        data: Batch,
        step: int,
        loss: torch.Tensor,
        outputs: torch.Tensor,
    ) -> None:
        if self.eval_progress.num_steps_completed_in_epoch == 1:
            tc = unittest.TestCase()
            recorded_timer_keys = none_throws(state.timer).recorded_durations.keys()
            for k in (
                "TimingAutoUnit.on_eval_start",
                "TimingAutoUnit.on_eval_epoch_start",
                "evaluate.iter(dataloader)",
                "evaluate.next(data_iter)",
                "TimingAutoUnit.move_data_to_device",
                "TimingAutoUnit.compute_loss",
                "TimingAutoUnit.on_eval_step_end",
            ):
                tc.assertIn(k, recorded_timer_keys)

            # eval_step should not be in the timer's recorded_durations because it overlaps with other timings in the AutoUnit's eval_step
            tc.assertNotIn("TimingAutoUnit.eval_step", recorded_timer_keys)

    def on_predict_step_end(
        self,
        state: State,
        data: Batch,
        step: int,
        outputs: torch.Tensor,
    ) -> None:
        if self.predict_progress.num_steps_completed_in_epoch == 1:
            tc = unittest.TestCase()
            recorded_timer_keys = none_throws(state.timer).recorded_durations.keys()
            for k in (
                "TimingAutoUnit.on_predict_start",
                "TimingAutoUnit.on_predict_epoch_start",
                "predict.iter(dataloader)",
                "predict.next(data_iter)",
                "TimingAutoUnit.move_data_to_device",
                "TimingAutoUnit.on_predict_step_end",
            ):
                tc.assertIn(k, recorded_timer_keys)

            # eval_step should not be in the timer's recorded_durations because it overlaps with other timings in the AutoUnit's eval_step
            tc.assertNotIn("TimingAutoUnit.predict_step", recorded_timer_keys)


class TimingAutoPredictUnit(AutoPredictUnit[Batch]):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__(module=module)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def compute_loss(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss, outputs

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
        my_optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        my_lr_scheduler = MagicMock()
        return my_optimizer, my_lr_scheduler

    def on_predict_step_end(
        self,
        state: State,
        data: TPredictData,
        step: int,
        outputs: torch.Tensor,
    ) -> None:
        if self.predict_progress.num_steps_completed_in_epoch == 1:
            tc = unittest.TestCase()
            recorded_timer_keys = none_throws(state.timer).recorded_durations.keys()
            for k in (
                "AutoPredictUnit.on_predict_start",
                "AutoPredictUnit.on_predict_epoch_start",
                "predict.iter(dataloader)",
                "AutoPredictUnit.next(data_iter)",
                "AutoPredictUnit.move_data_to_device",
                "AutoPredictUnit.forward",
                "AutoPredictUnit.on_predict_step_end",
            ):
                tc.assertIn(k, recorded_timer_keys)

            # predict_step should not be in the timer's recorded_durations because it overlaps with other timings in the AutoUnit's predict_step
            tc.assertNotIn("AutoPredictUnit.predict_step", recorded_timer_keys)
