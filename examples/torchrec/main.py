#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import sys
import tempfile
import uuid
from typing import Iterator, List, Optional, Tuple

import torch
from pyre_extensions import none_throws
from torch import distributed as dist
from torch.distributed import launcher
from torch.distributed.optim.apply_optimizer_in_backward import (
    _apply_optimizer_in_backward,
)
from torch.utils.data import DataLoader
from torcheval.metrics.classification.auroc import BinaryAUROC
from torcheval.metrics.toolkit import sync_and_compute
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.datasets.random import RandomRecDataset
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.models.dlrm import DLRM, DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchtnt.framework.callbacks import TQDMProgressBar
from torchtnt.framework.fit import fit
from torchtnt.framework.state import State
from torchtnt.framework.unit import EvalUnit, TrainUnit
from torchtnt.utils import (
    get_process_group_backend_from_device,
    init_from_env,
    rank_zero_print,
)
from torchtnt.utils.loggers.tensorboard import TensorBoardLogger


MIN_NODES = 1
MAX_NODES = 1
PROC_PER_NODE = 1


def init_dataloader(
    batch_size: int,
    num_batches: int,
    num_embeddings: int,
    backend: str,
    num_embeddings_per_feature: Optional[List[int]] = None,
    seed: Optional[int] = None,
    pin_memory: Optional[bool] = None,
) -> DataLoader:

    pin_memory = (backend == "nccl") if pin_memory is None else pin_memory
    return DataLoader(
        RandomRecDataset(
            keys=DEFAULT_CAT_NAMES,
            batch_size=batch_size,
            hash_size=num_embeddings,
            hash_sizes=num_embeddings_per_feature,
            manual_seed=seed,
            ids_per_feature=1,
            num_dense=len(DEFAULT_INT_NAMES),
            num_batches=num_batches,
        ),
        batch_size=None,
        batch_sampler=None,
        pin_memory=pin_memory,
        num_workers=0,
    )


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=10,
        help="batch size to use for validation and testing",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,64",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory when loading data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate.",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="log every n steps",
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_decay_start", type=int, default=0)
    parser.add_argument("--lr_decay_steps", type=int, default=0)

    parser.set_defaults(
        pin_memory=None,
    )
    return parser.parse_args(argv)


Batch = Tuple[torch.Tensor, torch.Tensor]


class MyUnit(TrainUnit[Iterator[Batch]], EvalUnit[Iterator[Batch]]):
    def __init__(
        self,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        tb_logger: TensorBoardLogger,
        train_auroc: BinaryAUROC,
        log_every_n_steps: int,
    ) -> None:
        super().__init__()
        self.module = module
        self.pipeline: TrainPipelineSparseDist = TrainPipelineSparseDist(
            module, optimizer, device, execute_all_batches=True
        )
        self.optimizer = optimizer
        self.device = device
        self.train_auroc = train_auroc
        self.tb_logger = tb_logger
        self.log_every_n_steps = log_every_n_steps

    def train_step(self, state: State, data: Iterator[Batch]) -> None:
        step = self.train_progress.num_steps_completed
        loss, logits, labels = self.pipeline.progress(data)
        preds = torch.sigmoid(logits)
        self.train_auroc.update(preds, labels)
        if step % self.log_every_n_steps == 0:
            accuracy = sync_and_compute(self.train_auroc)
            self.tb_logger.log("train_auroc", accuracy, step)
            self.tb_logger.log("loss", loss, step)

    def on_train_epoch_end(self, state: State) -> None:
        super().on_train_epoch_end(state)
        # reset the metric every epoch
        self.train_auroc.reset()

    def eval_step(self, state: State, data: Iterator[Batch]) -> None:
        step = self.eval_progress.num_steps_completed
        loss, _, _ = self.pipeline.progress(data)
        if step % self.log_every_n_steps == 0:
            self.tb_logger.log("evaluation_loss", loss, step)


def init_embdedding_configs(
    embedding_dim: int,
    num_embeddings_per_feature: Optional[List[int]],
    num_embeddings: int,
) -> List[EmbeddingBagConfig]:

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=embedding_dim,
            num_embeddings=(
                none_throws(num_embeddings_per_feature)[feature_idx]
                if num_embeddings is None
                else num_embeddings
            ),
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]
    return eb_configs


def init_model(
    eb_configs: List[EmbeddingBagConfig],
    dense_arch_layer_sizes: Optional[List[int]],
    over_arch_layer_sizes: Optional[List[int]],
    learning_rate: float,
    batch_size: int,
    device: torch.device,
) -> torch.nn.Module:

    dlrm_model = DLRM(
        embedding_bag_collection=EmbeddingBagCollection(
            tables=eb_configs, device=torch.device("meta")
        ),
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=dense_arch_layer_sizes or [],
        over_arch_layer_sizes=over_arch_layer_sizes or [],
        dense_device=device,
    )

    train_model = DLRMTrain(dlrm_model)

    _apply_optimizer_in_backward(
        torch.optim.SGD,
        train_model.model.sparse_arch.parameters(),
        {"lr": learning_rate},
    )

    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=get_local_size(),
            world_size=dist.get_world_size(),
            compute_device=device.type,
        ),
        batch_size=batch_size,
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
    )
    plan = planner.collective_plan(
        train_model, get_default_sharders(), dist.GroupMember.WORLD
    )

    model = DistributedModelParallel(
        module=train_model,
        device=device,
        plan=plan,
    )

    return model


def init_optimizer(
    model: torch.nn.Module, learning_rate: float
) -> torch.optim.Optimizer:
    optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(model.named_parameters())),
        lambda params: torch.optim.SGD(params, lr=learning_rate),
    )
    return optimizer


def init_logger() -> TensorBoardLogger:
    path = tempfile.mkdtemp()
    tb_logger = TensorBoardLogger(path)
    return tb_logger


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    for name, val in vars(args).items():
        try:
            vars(args)[name] = list(map(int, val.split(",")))
        except (ValueError, AttributeError):
            pass

    rank_zero_print(
        f"PARAMS: (lr, batch_size, warmup_steps, decay_start, decay_steps): {(args.learning_rate, args.batch_size, args.lr_warmup_steps, args.lr_decay_start, args.lr_decay_steps)}"
    )
    device = init_from_env()
    backend = get_process_group_backend_from_device(device)

    eb_configs = init_embdedding_configs(
        args.embedding_dim, args.num_embeddings_per_feature, args.num_embeddings
    )

    model = init_model(
        eb_configs,
        args.dense_arch_layer_sizes,
        args.over_arch_layer_sizes,
        args.learning_rate,
        args.batch_size,
        device,
    )

    optimizer = init_optimizer(model, args.learning_rate)
    tb_logger = init_logger()
    auroc = BinaryAUROC(device=device)

    my_unit = MyUnit(
        module=model,
        optimizer=optimizer,
        device=device,
        tb_logger=tb_logger,
        train_auroc=auroc,
        log_every_n_steps=10,
    )

    train_dataloader = init_dataloader(
        args.batch_size,
        args.num_batches,
        args.num_embeddings,
        backend,
        args.num_embeddings_per_feature,
        args.seed,
        args.pin_memory,
    )
    eval_dataloader = init_dataloader(
        args.test_batch_size,
        args.num_batches,
        args.num_embeddings,
        backend,
        args.num_embeddings_per_feature,
        args.seed,
        args.pin_memory,
    )

    tqdm_callback = TQDMProgressBar()
    fit(
        my_unit,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_epochs=args.epochs,
        callbacks=[tqdm_callback],
    )


if __name__ == "__main__":
    lc = launcher.LaunchConfig(
        min_nodes=MIN_NODES,
        max_nodes=MAX_NODES,
        nproc_per_node=PROC_PER_NODE,
        run_id=str(uuid.uuid4()),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        max_restarts=0,
        monitor_interval=1,
    )

    launcher.elastic_launch(config=lc, entrypoint=main)(sys.argv[1:])
