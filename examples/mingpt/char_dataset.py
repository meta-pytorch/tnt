# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import Dict, Tuple

import fsspec
import torch
from torch.utils.data import Dataset

"""
Adapted from https://github.com/karpathy/minGPT/blob/master/projects/chargpt/chargpt.py
"""


@dataclass
class DataConfig:
    path: str
    block_size: int
    train_split: float
    truncate: float = 1.0


class CharDataset(Dataset):
    def __init__(self, data_cfg: DataConfig) -> None:
        print(data_cfg.path)
        data = fsspec.open(data_cfg.path).open().read().decode("utf-8")
        data = data[: int(len(data) * data_cfg.truncate)]

        chars = sorted(set(data))
        data_size, vocab_size = len(data), len(chars)
        print("Data has %d characters, %d unique." % (data_size, vocab_size))

        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        self.block_size: int = data_cfg.block_size
        self.vocab_size: int = vocab_size
        self.data: str = data

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    # pyrefly: ignore [bad-override-param-name]
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx : idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
