# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import numpy as np
import torch
from torchtnt.utils.loggers.utils import scalar_to_float


class TestUtilities(unittest.TestCase):
    def test_scalar_to_float(self) -> None:
        invalid_tensor = torch.Tensor([1, 2, 3])
        with self.assertRaises(ValueError):
            scalar_to_float(invalid_tensor)

        float_x = 3.45
        valid_tensor = torch.Tensor([float_x])
        self.assertAlmostEqual(scalar_to_float(valid_tensor), float_x)

        invalid_ndarray = np.array([23.45, 15.21])
        with self.assertRaises(ValueError):
            scalar_to_float(invalid_ndarray)

        valid_ndarray = np.array([[[float_x]]])
        self.assertAlmostEqual(scalar_to_float(valid_ndarray), float_x)

    def test_scalar_to_float_bf16(self) -> None:
        float_x = 3.45
        valid_tensor = torch.Tensor([float_x]).to(torch.bfloat16)
        self.assertAlmostEqual(scalar_to_float(valid_tensor), float_x, delta=0.01)
