# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from numpy import ndarray
from torch import Tensor
from torchtnt.utils.loggers.logger import Scalar


def scalar_to_float(scalar: Scalar) -> float:
    if isinstance(scalar, Tensor):
        scalar = scalar.squeeze()
        numel = scalar.numel()
        if numel != 1:
            raise ValueError(
                f"Scalar tensor must contain a single item, {numel} given."
            )

        return float(scalar.cpu().detach().float().numpy().item())
    elif isinstance(scalar, ndarray):
        numel = scalar.size
        if numel != 1:
            raise ValueError(
                f"Scalar ndarray must contain a single item, {numel} given."
            )
        return float(scalar.item())

    return float(scalar)
