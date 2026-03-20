#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset
from torchtnt.framework.auto_unit import AutoUnit
from torchtnt.framework.state import EntryPoint, PhaseState, State
from torchtnt.framework.unit import EvalUnit, PredictUnit, TestUnit, TrainUnit
from torchtnt.utils.lr_scheduler import TLRScheduler

Batch = Tuple[torch.Tensor, torch.Tensor]


def get_dummy_train_state(dataloader: Optional[Iterable[object]] = None) -> State:
    return State(
        entry_point=EntryPoint.TRAIN,
        train_state=PhaseState(
            dataloader=dataloader or [1, 2, 3, 4],
            max_epochs=1,
            max_steps=1,
            max_steps_per_epoch=1,
        ),
        timer=None,
    )


def get_dummy_eval_state(dataloader: Optional[Iterable[object]] = None) -> State:
    return State(
        entry_point=EntryPoint.EVALUATE,
        eval_state=PhaseState(
            dataloader=dataloader or [1, 2, 3, 4],
            max_epochs=1,
            max_steps=1,
            max_steps_per_epoch=1,
        ),
        timer=None,
    )


def get_dummy_predict_state(dataloader: Optional[Iterable[object]] = None) -> State:
    return State(
        entry_point=EntryPoint.PREDICT,
        predict_state=PhaseState(
            dataloader=dataloader or [1, 2, 3, 4],
            max_epochs=1,
            max_steps=1,
            max_steps_per_epoch=1,
        ),
        timer=None,
    )


def get_dummy_test_state(dataloader: Optional[Iterable[object]] = None) -> State:
    return State(
        entry_point=EntryPoint.TEST,
        test_state=PhaseState(
            dataloader=dataloader or [1, 2, 3, 4],
            max_epochs=1,
            max_steps=1,
            max_steps_per_epoch=1,
        ),
        timer=None,
    )


def get_dummy_fit_state() -> State:
    return State(
        entry_point=EntryPoint.FIT,
        train_state=PhaseState(
            dataloader=[1, 2, 3, 4],
            max_epochs=1,
            max_steps=1,
            max_steps_per_epoch=1,
        ),
        eval_state=PhaseState(
            dataloader=[1, 2, 3, 4],
            max_epochs=1,
            max_steps=1,
            max_steps_per_epoch=1,
        ),
        timer=None,
    )


class DummyEvalUnit(EvalUnit[Batch]):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # initialize module & loss_fn
        self.module = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def eval_step(self, state: State, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss, outputs


class DummyPredictUnit(PredictUnit[Batch]):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # initialize module
        self.module = nn.Linear(input_dim, 2)

    def predict_step(self, state: State, data: Batch) -> torch.Tensor:
        inputs, targets = data

        outputs = self.module(inputs)
        return outputs


class DummyTestUnit(TestUnit[Batch]):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # initialize module & loss_fn
        self.module = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def test_step(self, state: State, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss, outputs


class DummyTrainUnit(TrainUnit[Batch]):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # initialize module, loss_fn, & optimizer
        self.module = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.01)

    def train_step(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss, outputs


class DummyMultiOptimUnit(TrainUnit[Batch]):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # initialize module, loss_fn, & optimizer

        self.modules: List[nn.Module] = [nn.Linear(input_dim, 2) for _ in range(6)]
        self.loss_fn = nn.CrossEntropyLoss()
        self.optims = [
            torch.optim.SGD,
            torch.optim.Adam,
            torch.optim.AdamW,
            torch.optim.Adadelta,
            torch.optim.NAdam,
            torch.optim.RMSprop,
        ]
        self.applied_optims: List[torch.optim.Optimizer] = []
        for module, optim in zip(self.modules, self.optims):
            o = optim(module.parameters(), lr=0.1)
            self.applied_optims.append(o)
            setattr(self, f"optimizer_{optim.__name__}", o)

    def train_step(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = [module(inputs) for module in self.modules]
        losses = [self.loss_fn(output, targets) for output in outputs]
        loss = torch.stack(losses).sum()
        loss.backward()

        for optim in self.applied_optims:
            optim.step()
            optim.zero_grad()

        return loss, outputs[0]


class DummyFitUnit(TrainUnit[Batch], EvalUnit[Batch]):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # initialize module, loss_fn, & optimizer
        self.module = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.01)

    def train_step(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss, outputs

    def eval_step(self, state: State, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss, outputs


class DummyFitTestUnit(TrainUnit[Batch], EvalUnit[Batch], TestUnit[Batch]):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # initialize module, loss_fn, & optimizer
        self.module = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.module.parameters(), lr=0.01, foreach=True
        )

    def train_step(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss, outputs

    def eval_step(self, state: State, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss, outputs

    def test_step(self, state: State, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss, outputs


def generate_random_dataset(num_samples: int, input_dim: int) -> Dataset[Batch]:
    """Returns a dataset of random inputs and labels for binary classification."""
    data = torch.randn(num_samples, input_dim)
    labels = torch.randint(low=0, high=2, size=(num_samples,))
    return TensorDataset(data, labels)


def generate_random_dataloader(
    num_samples: int, input_dim: int, batch_size: int
) -> DataLoader:
    return DataLoader(
        generate_random_dataset(num_samples, input_dim),
        batch_size=batch_size,
    )


class RandomIterableDataset(IterableDataset):
    def __init__(self, size: int, count: int) -> None:
        self.count: int = count
        self.size: int = size

    def __iter__(self) -> Iterator[Tensor]:
        for _ in range(self.count):
            yield torch.randn(self.size)

    def __len__(self) -> int:
        return self.count


def generate_random_iterable_dataloader(
    num_samples: int, input_dim: int, batch_size: int
) -> DataLoader:
    return DataLoader(
        dataset=RandomIterableDataset(input_dim, num_samples),
        batch_size=batch_size,
    )


class DummyAutoUnit(AutoUnit[Batch]):
    def compute_loss(self, state: State, data: Batch) -> Tuple[torch.Tensor, object]:
        inputs, targets = data
        outputs = self.module(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        return loss, outputs

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
        my_optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            my_optimizer, gamma=0.9
        )
        return my_optimizer, my_lr_scheduler


class DummyStatefulDataLoader:
    """Dummy Dataloader that implements state_dict and load_state_dict"""

    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader

    def state_dict(self) -> Dict[str, Any]:
        return {"current_batch": 1}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        return None

    def __iter__(self) -> Iterator[object]:
        return iter(self.dataloader)


def generate_dummy_stateful_dataloader(
    num_samples: int, input_dim: int, batch_size: int
) -> DummyStatefulDataLoader:
    return DummyStatefulDataLoader(
        DataLoader(
            dataset=RandomIterableDataset(input_dim, num_samples),
            batch_size=batch_size,
        )
    )


def generate_tensor_dataloader(
    samples: torch.Tensor, batch_size: int
) -> DummyStatefulDataLoader:
    return DummyStatefulDataLoader(
        DataLoader(
            dataset=TensorDataset(samples),
            batch_size=batch_size,
        )
    )


class DummyMeanMetric:
    def __init__(self) -> None:
        super().__init__()
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, value: float) -> None:
        self.sum += value
        self.count += 1

    def compute(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {"sum": self.sum, "count": self.count}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.sum = state_dict["sum"]
        self.count = state_dict["count"]


class DummyStatefulConfig:
    def __init__(
        self,
        storage_path: str,
        lazy_loading: bool,
        num_workers_per_gpu: int,
        max_batch_length: int,
    ) -> None:
        self.storage_path = storage_path
        self.lazy_loading = lazy_loading
        self.num_workers_per_gpu = num_workers_per_gpu
        self.max_batch_length = max_batch_length

    def state_dict(self) -> Dict[str, Any]:
        return {
            "storage_path": self.storage_path,
            "data": {
                "lazy_loading": self.lazy_loading,
                "train": {
                    "num_workers_per_gpu": self.num_workers_per_gpu,
                    "dynamic_batch_config": {
                        "max_batch_length": self.max_batch_length,
                    },
                },
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.storage_path = state_dict["storage_path"]
        self.lazy_loading = state_dict["data"]["lazy_loading"]
        self.num_workers_per_gpu = state_dict["data"]["train"]["num_workers_per_gpu"]
        self.max_batch_length = state_dict["data"]["train"]["dynamic_batch_config"][
            "max_batch_length"
        ]
