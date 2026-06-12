#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import logging
import tempfile
import uuid
from argparse import Namespace
from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed import launcher as pet
from torch.utils.data.dataset import Dataset, TensorDataset
from torcheval.metrics import BinaryAccuracy
from torchtnt.framework.auto_unit import AutoUnit, Strategy, SWAParams, TrainStepResults
from torchtnt.framework.fit import fit
from torchtnt.framework.state import EntryPoint, State
from torchtnt.utils import init_from_env, seed, TLRScheduler
from torchtnt.utils.loggers import TensorBoardLogger
from torchtnt.utils.prepare_module import ActivationCheckpointParams, TorchCompileParams

_logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


Batch = Tuple[torch.Tensor, torch.Tensor]
NUM_PROCESSES = 2


def prepare_module(input_dim: int, device: torch.device) -> nn.Module:
    """
    Instantiate a nn.Module which will define the architecture of your model. If using a data parallel technique, wrap the module in DDP or FSDP.
    See https://pytorch.org/docs/stable/generated/torch.nn.Module.html for docs.
    """
    return nn.Linear(input_dim, 1, device=device)


def _generate_dataset(num_samples: int, input_dim: int) -> Dataset[Batch]:
    """Returns a dataset of random inputs and labels for binary classification."""
    # TODO: use datapipes/dataloaderV2
    data = torch.randn(num_samples, input_dim)
    labels = torch.randint(low=0, high=2, size=(num_samples,))
    # pyrefly: ignore [bad-return]
    return TensorDataset(data, labels)


def prepare_dataloader(
    num_samples: int, input_dim: int, batch_size: int, device: torch.device
) -> torch.utils.data.DataLoader:
    """Instantiate DataLoader"""
    # pin_memory enables faster host to GPU copies
    on_cuda = device.type == "cuda"
    return torch.utils.data.DataLoader(
        _generate_dataset(num_samples, input_dim),
        batch_size=batch_size,
        pin_memory=on_cuda,
    )


class MyUnit(AutoUnit[Batch]):
    def __init__(
        self,
        *,
        train_accuracy: BinaryAccuracy,
        eval_accuracy: BinaryAccuracy,
        log_every_n_steps: int,
        tb_logger: Optional[TensorBoardLogger] = None,
        module: torch.nn.Module,
        device: Optional[torch.device] = None,
        strategy: Optional[Union[Strategy, str]] = None,
        step_lr_interval: Literal["step", "epoch"] = "epoch",
        precision: Optional[Union[str, torch.dtype]] = None,
        gradient_accumulation_steps: int = 1,
        detect_anomaly: Optional[bool] = None,
        clip_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
        swa_params: Optional[SWAParams] = None,
        torch_compile_params: Optional[TorchCompileParams] = None,
        activation_checkpoint_params: Optional[ActivationCheckpointParams] = None,
        training: bool = True,
    ) -> None:
        super().__init__(
            module=module,
            device=device,
            strategy=strategy,
            step_lr_interval=step_lr_interval,
            precision=precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            detect_anomaly=detect_anomaly,
            clip_grad_norm=clip_grad_norm,
            clip_grad_value=clip_grad_value,
            swa_params=swa_params,
            torch_compile_params=torch_compile_params,
            activation_checkpoint_params=activation_checkpoint_params,
            training=training,
        )
        self.tb_logger = tb_logger
        # create accuracy metrics to compute the accuracy of training and evaluation
        self.train_accuracy = train_accuracy
        self.eval_accuracy = eval_accuracy
        self.log_every_n_steps = log_every_n_steps

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, Optional[TLRScheduler]]:
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return optimizer, lr_scheduler

    def compute_loss(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data
        # convert targets to float Tensor for binary_cross_entropy_with_logits
        targets = targets.float()
        outputs = self.module(inputs)
        outputs = torch.squeeze(outputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)

        return loss, outputs

    def on_train_step_end(
        self,
        state: State,
        data: Batch,
        step: int,
        results: TrainStepResults,
    ) -> None:
        loss, outputs = results.loss, results.outputs
        _, targets = data
        self.train_accuracy.update(outputs, targets)
        tb_logger = self.tb_logger

        if step % self.log_every_n_steps == 0 and tb_logger is not None:
            accuracy = self.train_accuracy.compute()
            tb_logger.log("train_accuracy", accuracy, step)
            tb_logger.log("train_loss", loss, step)

    def on_eval_step_end(
        self,
        state: State,
        data: Batch,
        step: int,
        loss: torch.Tensor,
        outputs: torch.Tensor,
    ) -> None:
        _, targets = data
        self.eval_accuracy.update(outputs, targets)

    def on_eval_end(self, state: State) -> None:
        if state.entry_point == EntryPoint.FIT:
            step = self.train_progress.num_steps_completed
        else:
            step = self.eval_progress.num_steps_completed
        accuracy = self.eval_accuracy.compute()

        tb_logger = self.tb_logger
        if tb_logger is not None:
            tb_logger.log("eval_accuracy", accuracy, step)

        self.eval_accuracy.reset()

    def on_train_epoch_end(self, state: State) -> None:
        super().on_train_epoch_end(state)
        # reset the metric every epoch
        self.train_accuracy.reset()


def main(args: Namespace) -> None:
    # seed the RNG for better reproducibility. see docs https://pytorch.org/docs/stable/notes/randomness.html
    seed(args.seed)

    # device and process group initialization
    device = init_from_env()

    path = tempfile.mkdtemp()
    tb_logger = TensorBoardLogger(path)

    module = prepare_module(args.input_dim, device)
    train_accuracy = BinaryAccuracy(device=device)
    eval_accuracy = BinaryAccuracy(device=device)

    my_unit = MyUnit(
        tb_logger=tb_logger,
        train_accuracy=train_accuracy,
        eval_accuracy=eval_accuracy,
        module=module,
        device=device,
        strategy="ddp",
        log_every_n_steps=args.log_every_n_steps,
        precision=args.precision,
        gradient_accumulation_steps=4,
        detect_anomaly=True,
        clip_grad_norm=1.0,
    )

    train_dataloader = prepare_dataloader(
        args.num_batches_per_epoch, args.input_dim, args.batch_size, device
    )
    eval_dataloader = prepare_dataloader(
        args.num_batches_per_epoch, args.input_dim, args.batch_size, device
    )

    fit(
        my_unit,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_epochs=args.max_epochs,
    )


def get_args() -> Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--input-dim", type=int, default=32, help="input dimension")
    parser.add_argument("--max-epochs", type=int, default=2, help="training epochs")
    parser.add_argument(
        "--num-batches-per-epoch",
        type=int,
        default=1024,
        help="number of batches per epoch",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--log-every-n-steps", type=int, default=10, help="log every n steps"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="fp16, bf16, or fp32",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args: Namespace = get_args()
    lc = pet.LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=NUM_PROCESSES,
        run_id=str(uuid.uuid4()),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        max_restarts=0,
        monitor_interval=1,
    )

    pet.elastic_launch(lc, entrypoint=main)(args)
