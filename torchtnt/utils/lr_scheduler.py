# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch.optim.lr_scheduler

# This PR exposes LRScheduler as a public class
# https://github.com/pytorch/pytorch/pull/88503
try:
    TLRScheduler = torch.optim.lr_scheduler.LRScheduler
except AttributeError:
    TLRScheduler = torch.optim.lr_scheduler._LRScheduler

__all__ = ["TLRScheduler"]
