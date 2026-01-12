# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import sys
import tempfile
from argparse import Namespace
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adadelta
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
from torchtnt.framework.auto_unit import AutoUnit, TrainStepResults
from torchtnt.framework.fit import fit
from torchtnt.framework.state import State
from torchtnt.utils import init_from_env, seed, TLRScheduler
from torchtnt.utils.loggers import TensorBoardLogger
from torchvision import datasets, transforms

Batch = Tuple[torch.Tensor, torch.Tensor]


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class MyUnit(AutoUnit[Batch]):
    def __init__(
        self,
        *,
        tb_logger: TensorBoardLogger,
        train_accuracy: MulticlassAccuracy,
        log_every_n_steps: int,
        lr: float,
        gamma: float,
        module: torch.nn.Module,
        device: torch.device,
        strategy: str,
        precision: Optional[str],
        gradient_accumulation_steps: int,
        detect_anomaly: bool,
        clip_grad_norm: float,
    ) -> None:
        super().__init__(
            module=module,
            device=device,
            strategy=strategy,
            precision=precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            detect_anomaly=detect_anomaly,
            clip_grad_norm=clip_grad_norm,
        )
        self.tb_logger = tb_logger
        self.lr = lr
        self.gamma = gamma

        # create an accuracy Metric to compute the accuracy of training
        self.train_accuracy = train_accuracy
        self.log_every_n_steps = log_every_n_steps

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, Optional[TLRScheduler]]:
        optimizer = Adadelta(module.parameters(), lr=self.lr)
        lr_scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)
        return optimizer, lr_scheduler

    def compute_loss(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data
        outputs = self.module(inputs)
        outputs = torch.squeeze(outputs)
        loss = torch.nn.functional.nll_loss(outputs, targets)

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
        if step % self.log_every_n_steps == 0:
            accuracy = self.train_accuracy.compute()
            self.tb_logger.log("accuracy", accuracy, step)
            self.tb_logger.log("loss", loss, step)

    def on_train_epoch_end(self, state: State) -> None:
        super().on_train_epoch_end(state)
        # reset the metric every epoch
        self.train_accuracy.reset()

    def on_eval_step_end(
        self,
        state: State,
        data: Batch,
        step: int,
        loss: torch.Tensor,
        outputs: torch.Tensor,
    ) -> None:
        if step % self.log_every_n_steps == 0:
            self.tb_logger.log("evaluation loss", loss, step)


def main(argv: List[str]) -> None:
    # parse command line arguments
    args = get_args(argv)

    # seed the RNG for better reproducibility. see docs https://pytorch.org/docs/stable/notes/randomness.html
    seed(args.seed)

    # device and process group initialization
    device = init_from_env()

    # avoid torch autocast exception
    if device.type == "mps":
        device = torch.device("cpu")

    path = tempfile.mkdtemp()
    tb_logger = TensorBoardLogger(path)

    on_cuda = device.type == "cuda"

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    eval_dataset = datasets.MNIST("../data", train=False, transform=transform)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, pin_memory=on_cuda
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.test_batch_size, pin_memory=on_cuda
    )

    module = Net()
    train_accuracy = MulticlassAccuracy(device=device)

    my_unit = MyUnit(
        tb_logger=tb_logger,
        train_accuracy=train_accuracy,
        log_every_n_steps=args.log_every_n_steps,
        lr=args.lr,
        gamma=args.gamma,
        module=module,
        device=device,
        strategy="ddp",
        precision=args.precision,
        gradient_accumulation_steps=4,
        detect_anomaly=True,
        clip_grad_norm=1.0,
    )

    fit(
        my_unit,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_epochs=args.max_epochs,
        max_train_steps_per_epoch=args.max_train_steps_per_epoch,
    )

    if args.save_model:
        torch.save(module.state_dict(), "mnist_cnn.pt")


def get_args(argv: List[str]) -> Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )

    parser.add_argument(
        "--log-every-n-steps", type=int, default=10, help="log every n steps"
    )

    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="fp16 or bf16",
        choices=["fp16", "bf16"],
    )

    parser.add_argument(
        "--max-train-steps-per-epoch",
        type=int,
        default=20,
        help="the max number of steps to run per epoch for training",
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
