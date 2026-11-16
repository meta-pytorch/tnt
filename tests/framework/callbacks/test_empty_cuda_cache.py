#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from unittest import mock

from torchtnt.framework._test_utils import DummyFitUnit, generate_random_dataloader
from torchtnt.framework.callbacks.empty_cuda_cache import EmptyCudaCache
from torchtnt.framework.fit import fit


class EmptyCudaCacheTest(unittest.TestCase):
    def test_empty_cuda_cache_call_count_fit(self) -> None:
        """
        Test EmptyCudaCache callback was called correct number of times (with fit entry point)
        """
        input_dim = 2
        train_dataset_len = 10
        eval_dataset_len = 6
        batch_size = 2
        max_epochs = 2
        evaluate_every_n_epochs = 1
        expected_num_total_steps = (
            train_dataset_len / batch_size * max_epochs
            + eval_dataset_len / batch_size * max_epochs
        )
        step_interval = 4

        my_unit = DummyFitUnit(2)
        ecc_callback = EmptyCudaCache(step_interval)

        train_dataloader = generate_random_dataloader(
            train_dataset_len, input_dim, batch_size
        )
        eval_dataloader = generate_random_dataloader(
            eval_dataset_len, input_dim, batch_size
        )

        expected_num_calls_to_cuda_empty = expected_num_total_steps / step_interval
        with mock.patch(
            "torchtnt.framework.callbacks.empty_cuda_cache.torch.cuda.empty_cache"
        ) as empty_mock:
            fit(
                my_unit,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_epochs=max_epochs,
                evaluate_every_n_epochs=evaluate_every_n_epochs,
                callbacks=[ecc_callback],
            )
            self.assertEqual(empty_mock.call_count, expected_num_calls_to_cuda_empty)
