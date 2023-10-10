#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from typing import Optional
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import torch
import torch.distributed as dist
import torch.distributed.launcher as launcher
from pyre_extensions import none_throws
from torchtnt.utils.distributed import (
    _validate_global_rank_world_size,
    all_gather_tensors,
    destroy_process_group,
    get_file_init_method,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    get_process_group_backend_from_device,
    get_tcp_init_method,
    get_world_size,
    rank_zero_fn,
    revert_sync_batchnorm,
    sync_bool,
)
from torchtnt.utils.test_utils import get_pet_launch_config


class DistributedTest(unittest.TestCase):
    def test_get_process_group_backend_cpu(self) -> None:
        device = torch.device("cpu")
        pg_backend = get_process_group_backend_from_device(device)
        self.assertEqual(pg_backend, "gloo")

    def test_get_process_group_backend_gpu(self) -> None:
        device = torch.device("cuda:0")
        pg_backend = get_process_group_backend_from_device(device)
        self.assertEqual(pg_backend, "nccl")

    def test_get_world_size_single(self) -> None:
        self.assertEqual(get_world_size(), 1)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_get_world_size(self) -> None:
        world_size = 4
        config = get_pet_launch_config(world_size)
        launcher.elastic_launch(config, entrypoint=self._test_get_world_size)(
            world_size
        )

    @staticmethod
    def _test_get_world_size(world_size: int) -> None:
        assert get_world_size() == world_size

        dist.init_process_group("gloo")
        assert get_world_size() == dist.get_world_size()

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_get_global_rank(self) -> None:
        config = get_pet_launch_config(4)
        launcher.elastic_launch(config, entrypoint=self._test_get_global_rank)()

    @staticmethod
    def _test_get_global_rank() -> None:
        dist.init_process_group("gloo")
        assert get_global_rank() == dist.get_rank()

    def test_get_global_rank_single(self) -> None:
        self.assertEqual(get_global_rank(), 0)

    def test_get_local_rank_single(self) -> None:
        self.assertEqual(get_local_rank(), 0)
        self.assertEqual(get_local_world_size(), 1)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_get_local_rank(self) -> None:
        config = get_pet_launch_config(2)
        launcher.elastic_launch(config, entrypoint=self._test_get_local_rank)()

    @staticmethod
    def _test_get_local_rank() -> None:
        # when launched on a single node, these should be equal
        assert get_local_rank() == get_global_rank()
        assert get_local_world_size() == get_world_size()

    @staticmethod
    def _destroy_process_group() -> None:
        dist.init_process_group("gloo")
        destroy_process_group()
        assert not torch.distributed.is_initialized()

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_destroy_process_group(self) -> None:
        # should be a no-op if dist is not initialized
        destroy_process_group()
        config = get_pet_launch_config(2)
        launcher.elastic_launch(config, entrypoint=self._destroy_process_group)()

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_gather_uneven(self, world_size: Optional[int] = 4) -> None:
        config = get_pet_launch_config(2)
        launcher.elastic_launch(
            config, entrypoint=self._test_ddp_gather_uneven_tensors
        )()

    @staticmethod
    def _test_ddp_gather_uneven_tensors() -> None:
        dist.init_process_group("gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        tensor = torch.ones(rank)
        result = all_gather_tensors(tensor)
        assert len(result) == world_size
        for idx in range(world_size):
            assert len(result[idx]) == idx
            assert (result[idx] == torch.ones_like(result[idx])).all()

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_gather_uneven_multidim(self) -> None:
        config = get_pet_launch_config(2)
        launcher.elastic_launch(
            config, entrypoint=self._test_ddp_gather_uneven_tensors_multidim
        )()

    @staticmethod
    def _test_ddp_gather_uneven_tensors_multidim() -> None:
        dist.init_process_group("gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        tensor = torch.ones(rank + 1, 4 - rank)
        result = all_gather_tensors(tensor)
        assert len(result) == world_size
        for idx in range(world_size):
            val = result[idx]
            assert val.shape == (idx + 1, 4 - idx)
            assert (val == torch.ones_like(val)).all()

    def test_rank_zero_fn_rank_zero(self) -> None:
        @rank_zero_fn
        def foo() -> int:
            return 1

        x = foo()
        assert x == 1

    @patch("torchtnt.utils.distributed.get_global_rank")
    # pyre-fixme[2]: Parameter must be annotated.
    def test_rank_zero_fn_rank_non_zero(self, get_global_rank) -> None:
        get_global_rank.return_value = 1

        @rank_zero_fn
        def foo() -> int:
            return 1

        x = foo()
        assert x is None

    def test_revert_sync_batchnorm(self) -> None:
        original_batchnorm = torch.nn.modules.batchnorm.BatchNorm1d(4)
        # pyre-fixme[16]: `Optional` has no attribute `random_`.
        original_batchnorm.running_mean.random_(-1, 1)
        original_batchnorm.running_var.random_(0, 1)
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            original_batchnorm,
        )

        sync_bn_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        reverted_model = revert_sync_batchnorm(sync_bn_model)

        _, batch_norm = reverted_model.children()
        self.assertIsInstance(batch_norm, torch.nn.modules.batchnorm._BatchNorm)
        self.assertNotIsInstance(batch_norm, torch.nn.SyncBatchNorm)
        self.assertTrue(
            # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
            #  `Optional[Tensor]`.
            torch.equal(batch_norm.running_mean, original_batchnorm.running_mean)
        )
        self.assertTrue(
            # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
            #  `Optional[Tensor]`.
            torch.equal(batch_norm.running_var, original_batchnorm.running_var)
        )

    @classmethod
    def _full_sync_worker(cls, coherence_mode: Optional[str]) -> bool:
        dist.init_process_group("gloo")
        if dist.get_rank() == 0:
            val = True
        else:
            val = False
        # pyre-fixme[6]: For 2nd argument expected
        #  `Union[typing_extensions.Literal['all'], typing_extensions.Literal['any'],
        #  typing_extensions.Literal['rank_zero'], float, int]` but got
        #  `Optional[str]`.
        return sync_bool(val, coherence_mode=coherence_mode)

    def test_sync_bool_single_process(self) -> None:
        val = True
        new_val = sync_bool(val)
        # these should be the same in a single process case
        self.assertEqual(val, new_val)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_sync_bool_multi_process_coherence_mode_rank_zero(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(
            "rank_zero"
        )
        # Both processes should return True since rank 0 inputs True
        self.assertTrue(result[0])
        self.assertTrue(result[1])

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_sync_bool_multi_process_coherence_mode_any(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(
            "any"
        )
        # Both processes should return True since one of the processes inputs True
        self.assertTrue(result[0])
        self.assertTrue(result[1])

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_sync_bool_multi_process_coherence_mode_all(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(
            "all"
        )
        # Both processes should return False since not all processes input False
        self.assertFalse(result[0])
        self.assertFalse(result[1])

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_sync_bool_multi_process_coherence_mode_int_false(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(2)
        # Both processes should return False since 2 processes don't input True
        self.assertFalse(result[0])
        self.assertFalse(result[1])

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_sync_bool_multi_process_coherence_mode_int_true(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(1)
        # Both processes should return True since 1 processes inputs True
        self.assertTrue(result[0])
        self.assertTrue(result[1])

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_sync_bool_multi_process_coherence_mode_float_true(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(0.4)
        # Both processes should return True since 40% or one of the process inputs True
        self.assertTrue(result[0])
        self.assertTrue(result[1])

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_sync_bool_multi_process_coherence_mode_float_false(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(1.0)
        # Both processes should return False since 100% of processes don't input True
        self.assertFalse(result[0])
        self.assertFalse(result[1])

    def test_validate_global_rank_world_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid world_size value provided"):
            world_size = -1
            rank = 0
            _validate_global_rank_world_size(world_size=world_size, rank=rank)

        with self.assertRaisesRegex(ValueError, "Invalid rank value provided"):
            world_size = 2
            rank = -1
            _validate_global_rank_world_size(world_size=world_size, rank=rank)

        with self.assertRaisesRegex(
            ValueError, "Invalid rank and world_size values provided"
        ):
            world_size = 8
            rank = 8
            _validate_global_rank_world_size(world_size=world_size, rank=rank)

    def test_get_file_init_method(self) -> None:
        world_size = 10
        rank = 2
        my_filename = "/tmp/my_filename"
        init_method = get_file_init_method(
            world_size=world_size, rank=rank, filename=my_filename
        )
        url = urlparse(init_method)
        self.assertEqual(url.scheme, "file")
        self.assertEqual(url.netloc, "")
        self.assertEqual(url.path, my_filename)
        url_qs = parse_qs(url.query)
        self.assertIn("world_size", url_qs)
        self.assertEqual(url_qs["world_size"], [str(world_size)])
        self.assertIn("rank", url_qs)
        self.assertEqual(url_qs["rank"], [str(rank)])

        world_size = 2
        rank = 0
        # get temp filename
        init_method = get_file_init_method(
            world_size=world_size, rank=rank, filename=None
        )
        url = urlparse(init_method)
        self.assertEqual(url.scheme, "file")
        self.assertEqual(url.netloc, "")
        self.assertNotEqual(url.path, "")
        self.assertFalse(os.path.exists(url.path))
        url_qs = parse_qs(url.query)
        self.assertIn("world_size", url_qs)
        self.assertEqual(url_qs["world_size"], [str(world_size)])
        self.assertIn("rank", url_qs)
        self.assertEqual(url_qs["rank"], [str(rank)])

        world_size = 1
        rank = 0
        # get temp filename (default)
        init_method = get_file_init_method(world_size=world_size, rank=rank)
        url = urlparse(init_method)
        self.assertEqual(url.scheme, "file")
        self.assertEqual(url.netloc, "")
        self.assertNotEqual(url.path, "")
        self.assertFalse(os.path.exists(url.path))
        url_qs = parse_qs(url.query)
        self.assertIn("world_size", url_qs)
        self.assertEqual(url_qs["world_size"], [str(world_size)])
        self.assertIn("rank", url_qs)
        self.assertEqual(url_qs["rank"], [str(rank)])

    def test_get_tcp_init_method(self) -> None:
        world_size = 10
        rank = 2
        my_hostname = "my_hostname"
        my_port = 1234
        init_method = get_tcp_init_method(
            world_size=world_size, rank=rank, hostname=my_hostname, port=my_port
        )
        url = urlparse(init_method)
        self.assertEqual(url.scheme, "tcp")
        self.assertEqual(url.hostname, my_hostname)
        self.assertEqual(url.port, my_port)
        self.assertEqual(url.path, "")
        url_qs = parse_qs(url.query)
        self.assertIn("world_size", url_qs)
        self.assertEqual(url_qs["world_size"], [str(world_size)])
        self.assertIn("rank", url_qs)
        self.assertEqual(url_qs["rank"], [str(rank)])

        world_size = 2
        rank = 1
        my_hostname = "my_hostname"
        # get free port
        init_method = get_tcp_init_method(
            world_size=world_size, rank=rank, hostname=my_hostname, port=None
        )
        url = urlparse(init_method)
        self.assertEqual(url.scheme, "tcp")
        self.assertEqual(url.hostname, my_hostname)
        self.assertIsNotNone(url.port)
        self.assertTrue(none_throws(url.port) > 0)
        self.assertEqual(url.path, "")
        url_qs = parse_qs(url.query)
        self.assertIn("world_size", url_qs)
        self.assertEqual(url_qs["world_size"], [str(world_size)])
        self.assertIn("rank", url_qs)
        self.assertEqual(url_qs["rank"], [str(rank)])

        world_size = 12
        rank = 7
        my_port = 4321
        # get localhost
        init_method = get_tcp_init_method(
            world_size=world_size, rank=rank, hostname=None, port=my_port
        )
        url = urlparse(init_method)
        self.assertEqual(url.scheme, "tcp")
        self.assertIsNotNone(url.hostname)
        self.assertTrue(none_throws(url.hostname).startswith("localhost"))
        self.assertEqual(url.port, my_port)
        self.assertEqual(url.path, "")
        url_qs = parse_qs(url.query)
        self.assertIn("world_size", url_qs)
        self.assertEqual(url_qs["world_size"], [str(world_size)])
        self.assertIn("rank", url_qs)
        self.assertEqual(url_qs["rank"], [str(rank)])

        world_size = 128
        rank = 43
        # get localhost and free port
        init_method = get_tcp_init_method(world_size=world_size, rank=rank)
        url = urlparse(init_method)
        self.assertEqual(url.scheme, "tcp")
        self.assertIsNotNone(url.hostname)
        self.assertTrue(none_throws(url.hostname).startswith("localhost"))
        self.assertIsNotNone(url.port)
        self.assertTrue(none_throws(url.port) > 0)
        self.assertEqual(url.path, "")
        url_qs = parse_qs(url.query)
        self.assertIn("world_size", url_qs)
        self.assertEqual(url_qs["world_size"], [str(world_size)])
        self.assertIn("rank", url_qs)
        self.assertEqual(url_qs["rank"], [str(rank)])
