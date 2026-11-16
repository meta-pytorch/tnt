#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
import torch.distributed as dist
from torchtnt.utils.device import get_device_from_env
from torchtnt.utils.distributed import (
    all_gather_tensors,
    broadcast_str,
    get_global_rank,
    get_local_rank,
    PGWrapper,
    spawn_multi_process,
)
from torchtnt.utils.env import init_from_env
from torchtnt.utils.test_utils import skip_if_not_distributed, skip_if_not_gpu


class DistributedGPUTest(unittest.TestCase):
    @skip_if_not_gpu
    @skip_if_not_distributed
    def test_gather_uneven_multidim_nccl(self) -> None:
        spawn_multi_process(
            2,
            "nccl",
            self._test_ddp_gather_uneven_tensors_multidim_nccl,
        )

    @staticmethod
    def _test_ddp_gather_uneven_tensors_multidim_nccl() -> None:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        tensor = torch.ones(rank + 1, 4 - rank, device=get_device_from_env())
        result = all_gather_tensors(tensor)
        assert len(result) == world_size
        for idx in range(world_size):
            val = result[idx]
            assert val.shape == (idx + 1, 4 - idx)
            assert (val == 1).all()

    @skip_if_not_gpu
    @skip_if_not_distributed
    def test_pg_wrapper_scatter_object_list_nccl(self) -> None:
        spawn_multi_process(
            2,
            "nccl",
            self._test_pg_wrapper_scatter_object_list,
            timeout_s=180,
        )

    @classmethod
    def _test_pg_wrapper_scatter_object_list(
        cls,
    ) -> None:
        init_from_env()
        pg_wrapper = PGWrapper(dist.group.WORLD)
        output_list = [None] * 2
        pg_wrapper.scatter_object_list(
            output_list=output_list,
            input_list=[1, 2] if get_local_rank() == 0 else [None] * 2,
            src=0,
        )
        tc = unittest.TestCase()
        tc.assertEqual(output_list[0], get_local_rank() + 1)

    @staticmethod
    def _test_method(offset_arg: int, offset_kwarg: int) -> int:
        return get_global_rank() + offset_arg - offset_kwarg

    @skip_if_not_gpu
    @skip_if_not_distributed
    def test_spawn_multi_process(self) -> None:
        mp_list = spawn_multi_process(2, "nccl", self._test_method, 3, offset_kwarg=2)
        self.assertEqual(mp_list, [1, 2])

    @skip_if_not_gpu
    @skip_if_not_distributed
    def test_broadcast_str(self) -> None:
        spawn_multi_process(2, "gloo", self._test_broadcast_str)

    @skip_if_not_gpu
    @skip_if_not_distributed
    def test_broadcast_str_gpu(self) -> None:
        spawn_multi_process(2, "nccl", self._test_broadcast_str)

    @staticmethod
    def _test_broadcast_str() -> None:
        """
        Tests that test_broadcast_strworks as expected
        """
        init_from_env()
        val = None
        if dist.get_rank() == 0:
            val = "foo"

        broadcasted_val = broadcast_str(val)

        tc = unittest.TestCase()
        tc.assertEqual(broadcasted_val, "foo")
