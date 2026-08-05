#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest
from typing import Callable, cast, Literal, Optional, Union
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import torch
import torch.distributed as dist
import torch.distributed.launcher as launcher
from pyre_extensions import none_throws
from torch.distributed import ProcessGroup
from torchtnt.utils.distributed import (
    _validate_global_rank_world_size,
    all_gather_str,
    all_gather_tensors,
    broadcast_str,
    destroy_process_group,
    get_file_init_method,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    get_or_create_gloo_pg,
    get_process_group_backend_from_device,
    get_tcp_init_method,
    get_world_size,
    local_rank_zero_fn,
    PGWrapper,
    rank_zero_fn,
    rank_zero_read_and_broadcast,
    revert_sync_batchnorm,
    spawn_multi_process,
    sync_bool,
)
from torchtnt.utils.test_utils import get_pet_launch_config, skip_if_not_distributed


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

    @skip_if_not_distributed
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

    @skip_if_not_distributed
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

    @skip_if_not_distributed
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

    @skip_if_not_distributed
    def test_destroy_process_group(self) -> None:
        # should be a no-op if dist is not initialized
        destroy_process_group()
        config = get_pet_launch_config(2)
        launcher.elastic_launch(config, entrypoint=self._destroy_process_group)()

    @skip_if_not_distributed
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

    @skip_if_not_distributed
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
    def test_rank_zero_fn_rank_non_zero(self, get_global_rank: MagicMock) -> None:
        get_global_rank.return_value = 1

        @rank_zero_fn
        def foo() -> int:
            return 1

        x = foo()
        assert x is None

    def test_local_rank_zero_fn(self) -> None:
        @local_rank_zero_fn
        def foo() -> int:
            return 1

        x = foo()
        assert x == 1

    def test_revert_sync_batchnorm(self) -> None:
        original_batchnorm = torch.nn.modules.batchnorm.BatchNorm1d(4)

        none_throws(original_batchnorm.running_mean).random_(-1, 1)
        none_throws(original_batchnorm.running_var).random_(0, 1)
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
            torch.equal(
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                #  `Union[Tensor, Module]`.
                batch_norm.running_mean,
                none_throws(original_batchnorm.running_mean),
            )
        )
        self.assertTrue(
            torch.equal(
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                #  `Union[Tensor, Module]`.
                batch_norm.running_var,
                none_throws(original_batchnorm.running_var),
            )
        )

    @classmethod
    def _full_sync_worker(
        cls,
        coherence_mode: Union[Literal["any", "all", "rank_zero"], int, float] = "any",
    ) -> bool:
        dist.init_process_group("gloo")
        if dist.get_rank() == 0:
            val = True
        else:
            val = False
        return sync_bool(val, coherence_mode=coherence_mode)

    def test_sync_bool_single_process(self) -> None:
        val = True
        new_val = sync_bool(val)
        # these should be the same in a single process case
        self.assertEqual(val, new_val)

    @skip_if_not_distributed
    def test_sync_bool_multi_process_coherence_mode_rank_zero(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(
            "rank_zero"
        )
        # Both processes should return True since rank 0 inputs True
        self.assertTrue(result[0])
        self.assertTrue(result[1])

    @skip_if_not_distributed
    def test_sync_bool_multi_process_coherence_mode_any(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(
            "any"
        )
        # Both processes should return True since one of the processes inputs True
        self.assertTrue(result[0])
        self.assertTrue(result[1])

    @skip_if_not_distributed
    def test_sync_bool_multi_process_coherence_mode_all(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(
            "all"
        )
        # Both processes should return False since not all processes input False
        self.assertFalse(result[0])
        self.assertFalse(result[1])

    @skip_if_not_distributed
    def test_sync_bool_multi_process_coherence_mode_int_false(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(2)
        # Both processes should return False since 2 processes don't input True
        self.assertFalse(result[0])
        self.assertFalse(result[1])

    @skip_if_not_distributed
    def test_sync_bool_multi_process_coherence_mode_int_true(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(1)
        # Both processes should return True since 1 processes inputs True
        self.assertTrue(result[0])
        self.assertTrue(result[1])

    @skip_if_not_distributed
    def test_sync_bool_multi_process_coherence_mode_float_true(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(0.4)
        # Both processes should return True since 40% or one of the process inputs True
        self.assertTrue(result[0])
        self.assertTrue(result[1])

    @skip_if_not_distributed
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

    def test_pg_wrapper_scatter_object_list_gloo(self) -> None:
        config = get_pet_launch_config(4)
        launcher.elastic_launch(
            config, entrypoint=self._test_pg_wrapper_scatter_object_list
        )

    @classmethod
    def _test_pg_wrapper_scatter_object_list(
        cls,
    ) -> None:
        dist.init_process_group("gloo")
        pg_wrapper = PGWrapper(dist.group.WORLD)
        output_list = [None] * 4
        pg_wrapper.scatter_object_list(
            output_list=output_list,
            input_list=[1, 2, 3, 4] if get_local_rank() == 0 else [None] * 4,
            src=0,
        )
        tc = unittest.TestCase()
        tc.assertEqual(output_list[0], get_local_rank() + 1)

    @staticmethod
    def _test_method(offset_arg: int, offset_kwarg: int) -> int:
        return get_global_rank() + offset_arg - offset_kwarg

    @skip_if_not_distributed
    def test_spawn_multi_process(self) -> None:
        mp_list = spawn_multi_process(2, "gloo", self._test_method, 3, offset_kwarg=2)
        self.assertEqual(mp_list, [1, 2])

    @skip_if_not_distributed
    def test_rank_zero_read_and_broadcast(self) -> None:
        spawn_multi_process(2, "gloo", self._test_rank_zero_read_and_broadcast)

    @staticmethod
    def _test_rank_zero_read_and_broadcast() -> None:
        """
        Tests that rank_zero_read_and_broadcast decorator works as expected
        """

        @rank_zero_read_and_broadcast
        def _test_method_for_rank_zero() -> str:
            assert get_global_rank() == 0
            return "foo"

        val_from_test_method = _test_method_for_rank_zero()
        tc = unittest.TestCase()
        tc.assertEqual(val_from_test_method, "foo")

    @skip_if_not_distributed
    def test_get_or_create_gloo_pg(self) -> None:
        spawn_multi_process(2, "gloo", self._test_get_or_create_gloo_pg)

    @staticmethod
    @patch("torchtnt.utils.distributed.dist.destroy_process_group")
    @patch("torchtnt.utils.distributed.dist.barrier")
    def _test_get_or_create_gloo_pg(
        mock_barrier: MagicMock, mock_destroy_process_group: MagicMock
    ) -> None:

        # Get a side effect for the get_backend function that returns NCCL the first time it is called,
        # and then will return GLOO for subsequent calls. For use with _test_get_or_create_gloo_pg.
        def _get_backend_side_effect() -> Callable[[Optional[ProcessGroup]], str]:
            called_get_backend = False

            def get_backend(_) -> str:
                # We just want to return NCCL the first time we call this function.
                nonlocal called_get_backend
                if not called_get_backend:
                    called_get_backend = True
                    return dist.Backend.NCCL
                else:
                    return dist.Backend.GLOO  # real PG

            return get_backend

        tc = unittest.TestCase()
        cleanup_events = []
        mock_barrier.side_effect = lambda *, group: cleanup_events.append(
            ("barrier", group)
        )
        mock_destroy_process_group.side_effect = lambda group: cleanup_events.append(
            ("destroy", group)
        )

        # Test not distributed - no-op
        with patch(
            "torchtnt.utils.distributed.dist.is_initialized",
            return_value=False,
        ):
            with get_or_create_gloo_pg() as pg:
                tc.assertIsNone(pg)

        mock_destroy_process_group.assert_not_called()
        mock_barrier.assert_not_called()

        # Test no-op since gloo pg already exists
        mock_destroy_process_group.reset_mock()
        with get_or_create_gloo_pg() as pg:
            tc.assertIs(pg, dist.group.WORLD)

        mock_destroy_process_group.assert_not_called()
        mock_barrier.assert_not_called()

        # Test creating new gloo candidate pg - no op
        mock_destroy_process_group.reset_mock()
        gloo_pg = cast(ProcessGroup, dist.new_group(backend=dist.Backend.GLOO))
        with get_or_create_gloo_pg(gloo_pg) as pg:
            tc.assertIs(pg, gloo_pg)

        mock_destroy_process_group.assert_not_called()
        mock_barrier.assert_not_called()

        # Test with NCCL backend - should create a new gloo pg and destroy
        mock_destroy_process_group.reset_mock()

        with patch(
            "torchtnt.utils.distributed.dist.get_backend",
            side_effect=_get_backend_side_effect(),
        ):
            with get_or_create_gloo_pg() as pg:
                pg = none_throws(pg)
                tc.assertIsNot(pg, dist.group.WORLD)
                tc.assertEqual(pg._get_backend_name(), dist.Backend.GLOO)

        mock_barrier.assert_called_once_with(group=pg)
        mock_destroy_process_group.assert_called_once_with(pg)
        tc.assertEqual(cleanup_events, [("barrier", pg), ("destroy", pg)])

        # Test exception handling with existing pg - forward exception, group should not be destroyed
        mock_destroy_process_group.reset_mock()
        mock_barrier.reset_mock()
        cleanup_events.clear()
        with tc.assertRaisesRegex(Exception, "Test Exception"):
            gloo_pg = cast(ProcessGroup, dist.new_group(backend=dist.Backend.GLOO))
            with get_or_create_gloo_pg(gloo_pg) as pg:
                tc.assertIs(pg, gloo_pg)
                raise Exception("Test Exception")

        mock_destroy_process_group.assert_not_called()
        mock_barrier.assert_not_called()

        # Test exception handling with new pg - forward exception, group should be destroyed
        mock_destroy_process_group.reset_mock()
        mock_barrier.reset_mock()
        cleanup_events.clear()
        with tc.assertRaisesRegex(Exception, "Test Exception"):
            with patch(
                "torchtnt.utils.distributed.dist.get_backend",
                side_effect=_get_backend_side_effect(),
            ):
                with get_or_create_gloo_pg() as pg:
                    tc.assertIsNot(pg, dist.group.WORLD)
                    tc.assertEqual(
                        none_throws(pg)._get_backend_name(), dist.Backend.GLOO
                    )
                    raise Exception("Test Exception")

        mock_barrier.assert_called_once_with(group=pg)
        mock_destroy_process_group.assert_called_once_with(pg)
        tc.assertEqual(cleanup_events, [("barrier", pg), ("destroy", pg)])

        # Test that a cleanup barrier failure is propagated after the group is destroyed
        mock_destroy_process_group.reset_mock()
        mock_barrier.reset_mock()
        mock_barrier.side_effect = RuntimeError("Barrier cleanup failed")
        cleanup_events.clear()
        with tc.assertRaisesRegex(RuntimeError, "Barrier cleanup failed"):
            with patch(
                "torchtnt.utils.distributed.dist.get_backend",
                side_effect=_get_backend_side_effect(),
            ):
                with get_or_create_gloo_pg() as pg:
                    tc.assertIsNot(pg, dist.group.WORLD)
                    tc.assertEqual(
                        none_throws(pg)._get_backend_name(), dist.Backend.GLOO
                    )

        mock_barrier.assert_called_once_with(group=pg)
        mock_destroy_process_group.assert_called_once_with(pg)

    @skip_if_not_distributed
    def test_get_or_create_gloo_pg_real_cleanup(self) -> None:
        spawn_multi_process(2, "gloo", self._test_get_or_create_gloo_pg_real_cleanup)

    @staticmethod
    def _test_get_or_create_gloo_pg_real_cleanup() -> None:
        tc = unittest.TestCase()
        real_get_backend = dist.get_backend

        def get_backend(pg: Optional[ProcessGroup]) -> str:
            if pg is dist.group.WORLD:
                return dist.Backend.NCCL
            return real_get_backend(pg)

        with patch(
            "torchtnt.utils.distributed.dist.get_backend", side_effect=get_backend
        ):
            with get_or_create_gloo_pg() as pg:
                pg = none_throws(pg)
                value = torch.tensor([dist.get_rank() + 1])
                dist.all_reduce(value, group=pg)
                tc.assertEqual(value.item(), 3)

    @skip_if_not_distributed
    def test_broadcast_str_fixed_buffer_size(self) -> None:
        spawn_multi_process(2, "gloo", self._test_broadcast_str_fixed_buffer_size)

    @staticmethod
    def _test_broadcast_str_fixed_buffer_size() -> None:
        val = None
        if dist.get_rank() == 0:
            val = "foo"

        # Test case 1: fixed_buffer_size == len(val)
        broadcasted_val = broadcast_str(val, fixed_buffer_size=3)
        tc = unittest.TestCase()
        tc.assertEqual(broadcasted_val, "foo")

        # Test case 2: fixed_buffer_size > len(val)
        broadcasted_val = broadcast_str(val, fixed_buffer_size=10)
        tc.assertEqual(broadcasted_val, "foo")

    @skip_if_not_distributed
    def test_broadcast_str(self) -> None:
        spawn_multi_process(2, "gloo", self._test_broadcast_str)

    @staticmethod
    def _test_broadcast_str() -> None:
        """
        Tests that test_broadcast_str works as expected
        """

        val = None
        if dist.get_rank() == 0:
            val = "foo"

        broadcasted_val = broadcast_str(val)

        tc = unittest.TestCase()
        tc.assertEqual(broadcasted_val, "foo")

    @skip_if_not_distributed
    def test_all_gather_str(self) -> None:
        backend = "gloo"
        if torch.cuda.is_available():
            backend = "nccl"

        spawn_multi_process(2, backend, self._test_all_gather_str)

    @staticmethod
    def _test_all_gather_str() -> None:
        if torch.cuda.is_available():
            torch.cuda.set_device(dist.get_rank())

        val = None
        if dist.get_rank() == 0:
            val = "foo"
        else:
            val = "barzoo"

        # Test case 1: fixed_buffer_size == len(val)
        vals = all_gather_str(val)

        tc = unittest.TestCase()
        tc.assertEqual(vals[0], "foo")
        tc.assertEqual(vals[1], "barzoo")
