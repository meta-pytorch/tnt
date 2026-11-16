# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from torchtnt.utils.data.iterators import StoppingMechanism


class TestIterators(unittest.TestCase):

    def test_stopping_mechanism_comparison(self) -> None:
        self.assertTrue(
            StoppingMechanism.ALL_DATASETS_EXHAUSTED == "ALL_DATASETS_EXHAUSTED"
        )
        self.assertTrue(
            StoppingMechanism.ALL_DATASETS_EXHAUSTED
            == StoppingMechanism.ALL_DATASETS_EXHAUSTED
        )
        self.assertFalse(
            StoppingMechanism.ALL_DATASETS_EXHAUSTED == "SMALLEST_DATASET_EXHAUSTED"
        )
        self.assertFalse(
            StoppingMechanism.ALL_DATASETS_EXHAUSTED
            == StoppingMechanism.SMALLEST_DATASET_EXHAUSTED
        )
