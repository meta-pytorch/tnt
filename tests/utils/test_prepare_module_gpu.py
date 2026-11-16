#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecisionPolicy,
)
from torchtnt.framework._test_utils import DummyAutoUnit, generate_random_dataloader
from torchtnt.framework.train import train
from torchtnt.utils.distributed import get_global_rank
from torchtnt.utils.prepare_module import get_module_state_dict

try:
    from torch.distributed.fsdp import fully_shard
except ImportError:

    def noop(*args: Any, **kwargs: Any) -> None:
        pass

    fully_shard = noop

from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtnt.utils.distributed import spawn_multi_process
from torchtnt.utils.env import init_from_env
from torchtnt.utils.prepare_module import (
    _is_fsdp_module,
    DDPStrategy,
    FSDP2Strategy,
    FSDPStrategy,
    prepare_ddp,
    prepare_fsdp,
    prepare_fsdp2,
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
        spawn_multi_process(
            2,
            "nccl",
            self._test_prepare_ddp_meta_device,
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

    @staticmethod
    def _test_prepare_ddp_meta_device() -> None:
        module = torch.nn.Linear(2, 2, device="meta")
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
        model = FSDP(model)
        assert _is_fsdp_module(model)

        # test fsdp2
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

    @skip_if_not_distributed
    @skip_if_not_gpu
    def test_prepare_fsdp2(self) -> None:
        """
        Launch tests of FSDP2 strategy
        """

        spawn_multi_process(
            1,
            "nccl",
            self._test_prepare_fsdp2_none_sharded_raises,
        )

        spawn_multi_process(
            1,
            "nccl",
            self._test_prepare_fsdp2_shard_all,
        )

        spawn_multi_process(
            1,
            "nccl",
            self._test_prepare_fsdp2_submodule,
        )

        spawn_multi_process(
            1,
            "nccl",
            self._test_prepare_fsdp2_meta_device,
        )

    @staticmethod
    def _test_prepare_fsdp2_none_sharded_raises() -> None:
        """
        Test with a strategy that does not shard any modules, should raise error. And also raise error
        for invalid mp_policy.
        """
        tc = unittest.TestCase()

        module = SimpleModule()
        device = torch.device("cuda")
        strategy = FSDP2Strategy(modules_to_shard=[])
        with tc.assertRaises(ValueError):
            prepare_fsdp2(module, device, strategy)

        # pyre-ignore[6]: Incompatible parameter type (intentional for testing)
        strategy = FSDP2Strategy(mp_policy=MixedPrecisionPolicy(param_dtype=16))
        with tc.assertRaises(ValueError):
            prepare_fsdp2(module, device, strategy)

    @staticmethod
    def _test_prepare_fsdp2_shard_all() -> None:
        """
        Test with a strategy that shards all modules
        """
        tc = unittest.TestCase()

        module = SimpleModule()
        device = torch.device("cuda")
        strategy = FSDP2Strategy(modules_to_shard="all", mp_policy=torch.bfloat16)
        prepare_fsdp2(module, device, strategy)

        for submodule in module.modules():
            tc.assertTrue(_is_fsdp_module(submodule))

    @staticmethod
    def _test_prepare_fsdp2_submodule() -> None:
        """
        Test with a strategy that shards modules (either str or module type)
        """
        tc = unittest.TestCase()

        for t in (torch.nn.Linear, "Linear"):
            module = SimpleModule()
            device = torch.device("cuda")
            # pyre-ignore: Incompatible parameter type [6] (intentional for this test)
            mp_policy = MixedPrecisionPolicy(param_dtype="bf16")
            strategy = FSDP2Strategy(modules_to_shard=(t,), mp_policy=mp_policy)
            prepare_fsdp2(module, device, strategy)

            for submodule in module.modules():
                if isinstance(submodule, torch.nn.Conv2d):
                    tc.assertFalse(_is_fsdp_module(submodule))
                else:
                    # linear and SimpleModule are fsdp modules
                    tc.assertTrue(_is_fsdp_module(submodule))

    @staticmethod
    def _test_prepare_fsdp2_meta_device() -> None:
        """
        Test with a strategy that shards specific modules on meta device
        """
        tc = unittest.TestCase()

        module = SimpleModule(meta_device=True)
        device = torch.device("cuda")
        strategy = FSDP2Strategy(modules_to_shard=(torch.nn.Linear,))
        prepare_fsdp2(module, device, strategy)

        for submodule in module.modules():
            if isinstance(submodule, torch.nn.Conv2d):
                tc.assertFalse(_is_fsdp_module(submodule))
            else:
                # linear and SimpleModule are fsdp modules
                tc.assertTrue(_is_fsdp_module(submodule))

    def test_get_module_state_dict(self) -> None:
        spawn_multi_process(
            2,
            "nccl",
            self._test_get_module_state_dict,
        )

    @staticmethod
    def _test_get_module_state_dict() -> None:
        rank = get_global_rank()

        fsdp_strategy = FSDPStrategy(
            sharding_strategy="FULL_SHARD",
            auto_wrap_policy=lambda module, recurse, nonwrapped_numel: True,
        )
        ddp_strategy = DDPStrategy()

        for strategy, rank0_only in (
            (fsdp_strategy, True),
            (fsdp_strategy, False),
            (ddp_strategy, True),
            (ddp_strategy, False),
            (None, True),
            (None, False),
        ):
            module = torch.nn.Sequential(
                torch.nn.Linear(2, 100),
                torch.nn.Linear(100, 2),
            )

            unit = DummyAutoUnit(
                module=module,
                strategy=strategy,
            )

            dataloader = generate_random_dataloader(10, 2, 10)
            train(unit, dataloader, max_epochs=1)

            module_sd = get_module_state_dict(unit.module, rank0_only=rank0_only)

            tc = unittest.TestCase()

            # For FSDP, if the user passed rank0_only=True, we should get an empty state dict
            # on all ranks except rank 0
            if rank0_only and isinstance(strategy, FSDPStrategy) and rank != 0:
                tc.assertEqual(module_sd, {})

            else:
                # Make sure that the generated state dict has the actual model keys,
                # and the values are actual tensors as opposed to ShardedTensor.
                tc.assertCountEqual(
                    ["0.weight", "0.bias", "1.weight", "1.bias"],
                    list(module_sd.keys()),
                )
                tc.assertIsInstance(module_sd["0.weight"], torch.Tensor)


class SimpleModule(torch.nn.Module):
    def __init__(self, meta_device: bool = False) -> None:
        super(SimpleModule, self).__init__()
        self.linear = torch.nn.Linear(10, 10, device="meta" if meta_device else None)
        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            device="meta" if meta_device else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
