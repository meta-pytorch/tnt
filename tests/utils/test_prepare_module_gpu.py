#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest

import torch

from torch.distributed._composable import fully_shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtnt.utils.distributed import spawn_multi_process
from torchtnt.utils.env import init_from_env
from torchtnt.utils.prepare_module import (
    _is_fsdp_module,
    DDPStrategy,
    FSDPStrategy,
    prepare_ddp,
    prepare_fsdp,
    prepare_module,
)
from torchtnt.utils.test_utils import skip_if_not_distributed, skip_if_not_gpu


class PrepareModelGPUTest(unittest.TestCase):
    @skip_if_not_gpu
    def test_prepare_no_strategy(self) -> None:
        module = torch.nn.Linear(2, 2)  # initialize on cpu
        device = init_from_env()  # should be cuda device
        module = prepare_module(module, device, strategy=None)
        self.assertEqual(next(module.parameters()).device, device)

    @skip_if_not_gpu
    @skip_if_not_distributed
    def test_prepare_ddp(self) -> None:
        spawn_multi_process(
            2,
            "nccl",
            self._test_prepare_ddp,
        )

    @staticmethod
    def _test_prepare_ddp() -> None:
        module = torch.nn.Linear(2, 2)
        device = init_from_env()
        ddp_module = prepare_ddp(
            module,
            device,
            DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
        )
        tc = unittest.TestCase()
        tc.assertTrue(isinstance(ddp_module, DDP))

    @skip_if_not_gpu
    @skip_if_not_distributed
    def test_prepare_fsdp(self) -> None:
        spawn_multi_process(
            2,
            "nccl",
            self._test_prepare_fsdp,
        )

    @staticmethod
    def _test_prepare_fsdp() -> None:
        module = torch.nn.Linear(2, 2)
        device = init_from_env()
        fsdp_module = prepare_fsdp(module, device, FSDPStrategy(limit_all_gathers=True))
        tc = unittest.TestCase()
        tc.assertTrue(isinstance(fsdp_module, FSDP))

    @skip_if_not_distributed
    @unittest.skipUnless(
        condition=bool(torch.cuda.device_count() >= 2),
        reason="This test needs 2 GPUs to run.",
    )
    def test_is_fsdp_module(self) -> None:
        spawn_multi_process(
            2,
            "gloo",
            self._test_is_fsdp_module,
        )

    @staticmethod
    def _test_is_fsdp_module() -> None:
        device = init_from_env()
        model = torch.nn.Linear(1, 1, device=device)
        assert not _is_fsdp_module(model)
        model = FSDP(torch.nn.Linear(1, 1, device=device))
        assert _is_fsdp_module(model)
        model = torch.nn.Linear(1, 1, device=device)
        fully_shard(model)
        assert _is_fsdp_module(model)

    @skip_if_not_distributed
    @skip_if_not_gpu
    def test_fdsp_precision(self) -> None:
        spawn_multi_process(
            2,
            "nccl",
            self._test_fdsp_precision,
        )

    @staticmethod
    def _test_fdsp_precision() -> None:
        module = torch.nn.Linear(1, 1)
        device = init_from_env()
        mixed_precision = MixedPrecision(
            param_dtype=torch.float64,
        )
        fsdp_module = prepare_fsdp(
            module, device, FSDPStrategy(mixed_precision=mixed_precision)
        )
        tc = unittest.TestCase()
        tc.assertTrue(isinstance(fsdp_module, FSDP))
        tc.assertEqual(
            fsdp_module.mixed_precision.param_dtype, mixed_precision.param_dtype
        )

    @skip_if_not_distributed
    @skip_if_not_gpu
    @unittest.skipUnless(
        condition=bool(torch.cuda.device_count() >= 2),
        reason="This test needs 2 GPUs to run.",
    )
    def test_fdsp_str_types(self) -> None:
        spawn_multi_process(
            2,
            "nccl",
            self._test_fdsp_precision_str_types,
        )
        spawn_multi_process(
            2,
            "nccl",
            self._test_fdsp_backward_prefetch_str_types,
        )
        spawn_multi_process(
            2,
            "nccl",
            self._test_fdsp_sharding_strategy_str_types,
        )
        spawn_multi_process(
            2,
            "nccl",
            self._test_fdsp_state_dict_str_types,
        )

    @staticmethod
    def _test_fdsp_precision_str_types() -> None:
        from torchtnt.utils.prepare_module import MixedPrecision as _MixedPrecision

        module = torch.nn.Linear(1, 1)
        device = init_from_env()
        mixed_precision = _MixedPrecision(
            param_dtype="fp16",
            reduce_dtype="bf16",
            buffer_dtype="fp32",
        )

        fsdp_module = prepare_fsdp(
            module, device, FSDPStrategy(mixed_precision=mixed_precision)
        )
        tc = unittest.TestCase()
        tc.assertTrue(isinstance(fsdp_module, FSDP))

    @staticmethod
    def _test_fdsp_backward_prefetch_str_types() -> None:
        module = torch.nn.Linear(1, 1)
        device = init_from_env()

        tc = unittest.TestCase()
        for value in ["BACKWARD_PRE", "BACKWARD_POST"]:
            fsdp_module = prepare_fsdp(
                module, device, FSDPStrategy(backward_prefetch=value)
            )
            tc.assertTrue(isinstance(fsdp_module, FSDP), f"tested value: {value}")

    @staticmethod
    def _test_fdsp_sharding_strategy_str_types() -> None:
        module = torch.nn.Linear(1, 1)
        device = init_from_env()

        tc = unittest.TestCase()
        for value in [
            "FULL_SHARD",
            "SHARD_GRAD_OP",
            "NO_SHARD",
            # skip hybrid strategy; tricky to configure in-test
        ]:

            fsdp_module = prepare_fsdp(
                module,
                device,
                FSDPStrategy(sharding_strategy=value),
            )
            tc.assertTrue(isinstance(fsdp_module, FSDP), f"tested value: {value}")

    @staticmethod
    def _test_fdsp_state_dict_str_types() -> None:
        module = torch.nn.Linear(1, 1)
        device = init_from_env()

        tc = unittest.TestCase()
        for value in [
            "FULL_STATE_DICT",
            "LOCAL_STATE_DICT",
            "SHARDED_STATE_DICT",
        ]:
            fsdp_module = prepare_fsdp(
                module, device, FSDPStrategy(state_dict_type=value)
            )
            tc.assertTrue(isinstance(fsdp_module, FSDP), f"tested value: {value}")

    @skip_if_not_distributed
    @skip_if_not_gpu
    def test_prepare_module_with_fsdp(self) -> None:
        """
        Launch tests of FSDP strategy
        """
        spawn_multi_process(
            2,
            "nccl",
            self._test_prepare_module_fsdp_strategy_wrapped_in_fsdp,
        )
        spawn_multi_process(
            2,
            "nccl",
            self._test_prepare_module_fsdp_string_wrapped_in_fsdp,
        )

    @staticmethod
    def _test_prepare_module_fsdp_strategy_wrapped_in_fsdp() -> None:
        """
        Test that the module is correctly wrapped in FSDP
        """

        fsdp_module = prepare_module(
            module=torch.nn.Linear(2, 2),
            device=init_from_env(),
            strategy=FSDPStrategy(),
        )
        tc = unittest.TestCase()

        tc.assertTrue(isinstance(fsdp_module, FSDP))

    @staticmethod
    def _test_prepare_module_fsdp_string_wrapped_in_fsdp() -> None:
        """
        Test that the module is correctly wrapped in FSDP when passing "fsdp" as a string
        """

        fsdp_module = prepare_module(
            module=torch.nn.Linear(2, 2),
            device=init_from_env(),
            strategy="fsdp",
        )
        tc = unittest.TestCase()

        tc.assertTrue(isinstance(fsdp_module, FSDP))
