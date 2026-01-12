# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import logging
import tempfile
from argparse import Namespace
from typing import Optional, Tuple

import torch
from libfb.py import parutil
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset, Subset
from torchtnt.examples.mingpt.char_dataset import CharDataset, DataConfig
from torchtnt.examples.mingpt.model import (
    create_optimizer,
    GPT,
    GPTConfig,
    OptimizerConfig,
)
from torchtnt.framework.auto_unit import AutoUnit, TrainStepResults
from torchtnt.framework.fit import fit
from torchtnt.framework.state import State
from torchtnt.utils import init_from_env, seed, TLRScheduler
from torchtnt.utils.loggers import TensorBoardLogger

_logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

Batch = Tuple[torch.Tensor, torch.Tensor]
PATH: str = parutil.get_file_path("data/input.txt", pkg=__package__)


def prepare_dataloader(
    dataset: Dataset[CharDataset],
    batch_size: int,
    device: torch.device,
) -> torch.utils.data.DataLoader:
    """Instantiate DataLoader"""
    # pin_memory enables faster host to GPU copies
    on_cuda = device.type == "cuda"
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=on_cuda,
    )


def get_datasets(
    data_cfg: DataConfig,
) -> Tuple[Subset[CharDataset], Subset[CharDataset], CharDataset]:
    dataset = CharDataset(data_cfg)
    train_len = int(len(dataset) * data_cfg.train_split)
    train_set, eval_set = random_split(dataset, [train_len, len(dataset) - train_len])
    return train_set, eval_set, dataset


class MinGPTUnit(AutoUnit[Batch]):
    def __init__(
        self,
        tb_logger: TensorBoardLogger,
        opt_cfg: OptimizerConfig,
        log_every_n_steps: int,
        module: torch.nn.Module,
        device: torch.device,
        strategy: str,
        gradient_accumulation_steps: int,
        detect_anomaly: bool,
        clip_grad_norm: float,
    ) -> None:
        super().__init__(
            module=module,
            device=device,
            strategy=strategy,
            gradient_accumulation_steps=gradient_accumulation_steps,
            detect_anomaly=detect_anomaly,
            clip_grad_norm=clip_grad_norm,
        )
        self.tb_logger = tb_logger
        self.opt_cfg = opt_cfg
        self.log_every_n_steps = log_every_n_steps

    def configure_optimizers_and_lr_scheduler(
        self,
        module: torch.nn.Module,
    ) -> Tuple[torch.optim.Optimizer, Optional[TLRScheduler]]:
        optimizer = create_optimizer(module, self.opt_cfg)
        return optimizer, None

    def compute_loss(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input, target = data
        outputs, loss = self.module(input, target)
        return loss, outputs

    def on_train_step_end(
        self,
        state: State,
        data: Batch,
        step: int,
        results: TrainStepResults,
    ) -> None:
        loss = results.loss
        if step % self.log_every_n_steps == 0:
            self.tb_logger.log("loss", loss, step)


def main(args: Namespace) -> None:

    seed(args.seed)
    device = init_from_env()

    data_cfg = DataConfig(
        path=PATH,
        block_size=args.block_size,
        train_split=args.train_split,
        truncate=args.truncate,
    )
    train_data, eval_data, dataset = get_datasets(data_cfg)
    train_dataloader = prepare_dataloader(train_data, args.batch_size, device)
    eval_dataloader = prepare_dataloader(eval_data, args.batch_size, device)

    gpt_cfg = GPTConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        vocab_size=dataset.vocab_size,
        block_size=dataset.block_size,
        device=str(device),
    )
    module = GPT(gpt_cfg)

    path = tempfile.mkdtemp()
    tb_logger = TensorBoardLogger(path)

    my_unit = MinGPTUnit(
        tb_logger=tb_logger,
        opt_cfg=OptimizerConfig(learning_rate=args.lr, weight_decay=args.weight_decay),
        module=module,
        device=device,
        strategy="ddp" if torch.distributed.is_initialized() else None,
        log_every_n_steps=args.log_every_n_steps,
        gradient_accumulation_steps=4,
        detect_anomaly=True,
        clip_grad_norm=args.clip_grad_norm,
    )

    fit(
        my_unit,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        max_train_steps_per_epoch=args.max_steps_epoch,
    )


def get_args() -> Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--truncate", type=float, default=0.05, help="truncate data")
    parser.add_argument("--train-split", type=float, default=0.9, help="training split")
    parser.add_argument("--block-size", type=int, default=128, help="block size")
    parser.add_argument("--batch-size", type=int, default=216, help="batch size")
    parser.add_argument("--max-steps", type=int, help="training steps")
    parser.add_argument("--max-steps-epoch", type=int, help="training steps per epoch")
    parser.add_argument("--max-epochs", type=int, default=10, help="training epochs")
    parser.add_argument("--n-layer", type=int, default=8, help="number of block layers")
    parser.add_argument("--n-head", type=int, default=8, help="number of heads")
    parser.add_argument("--n-embd", type=int, default=512, help="embedding dimension")
    parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="weight decay")
    parser.add_argument(
        "--clip-grad-norm", type=float, default=1.0, help="clip gradient norm"
    )
    parser.add_argument(
        "--log-every-n-steps", type=int, default=10, help="log every n steps"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
