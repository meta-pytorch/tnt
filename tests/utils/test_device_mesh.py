# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from torchtnt.utils.device_mesh import (
    create_device_mesh,
    get_dp_local_rank,
    get_dp_mesh_size,
    GlobalMeshCoordinator,
)
from torchtnt.utils.distributed import get_global_rank, spawn_multi_process


class TestCreateDeviceMesh(unittest.TestCase):
    def test_create_device_mesh(
        self,
    ) -> None:
        spawn_multi_process(
            4,
            "gloo",
            self._test_create_device_mesh,
        )

    @staticmethod
    def _test_create_device_mesh() -> None:
        tc = unittest.TestCase()

        with tc.assertRaisesRegex(ValueError, "World size 4 must be divisible by"):
            create_device_mesh(dp_shard=-1, dp_replicate=1, tp=8, device_type="cpu")

        with tc.assertRaisesRegex(ValueError, "World size 4 must be divisible by"):
            create_device_mesh(dp_shard=-1, dp_replicate=1, tp=3, device_type="cpu")

        device_mesh = create_device_mesh(
            dp_shard=-1, dp_replicate=2, tp=None, device_type="cpu"
        )

        tc.assertEqual(device_mesh["dp_shard"].size(), 2)


class TestGlobalMeshCoordinator(unittest.TestCase):
    def test_attrs(self) -> None:
        spawn_multi_process(1, "gloo", self._test_attrs)

    @staticmethod
    def _test_attrs() -> None:
        """
        Test local attributes of GlobalMeshCoordinator are set correctly
        """
        tc = unittest.TestCase()

        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=None, device_type="cpu"
        )
        tc.assertFalse(gmc._dp_replicate_enabled)
        tc.assertFalse(gmc._tp_enabled)

        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=1, device_type="cpu"
        )
        tc.assertFalse(gmc._dp_replicate_enabled)
        tc.assertTrue(gmc._tp_enabled)

    def test_tp_mesh(self) -> None:
        spawn_multi_process(4, "gloo", self._test_tp_mesh)

    @staticmethod
    def _test_tp_mesh() -> None:
        """
        Test tp_mesh is returned correctly
        """
        tc = unittest.TestCase()

        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=None, device_type="cpu"
        )
        tc.assertIsNone(gmc.tp_mesh)

        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=4, device_type="cpu"
        )
        tc.assertIsNotNone(gmc.tp_mesh)
        tc.assertEqual(gmc.tp_mesh.size(), 4)

    def test_dp_mesh(self) -> None:
        spawn_multi_process(4, "gloo", self._test_dp_mesh)

    @staticmethod
    def _test_dp_mesh() -> None:
        """
        Test dp_mesh is returned correctly
        """
        tc = unittest.TestCase()

        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=None, device_type="cpu"
        )
        tc.assertEqual(gmc.dp_mesh, gmc.device_mesh["dp_shard"])
        tc.assertEqual(get_dp_mesh_size(gmc), 4)
        tc.assertEqual(get_dp_local_rank(gmc), get_global_rank())

        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=2, tp=None, device_type="cpu"
        )
        tc.assertEqual(gmc.dp_mesh, gmc.device_mesh["dp"])
        tc.assertEqual(get_dp_mesh_size(gmc), 4)
        tc.assertEqual(get_dp_local_rank(gmc), get_global_rank())

        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=2, device_type="cpu"
        )
        tc.assertEqual(gmc.dp_mesh, gmc.device_mesh["dp_shard"])
        tc.assertEqual(get_dp_mesh_size(gmc), 2)
        tc.assertEqual(get_dp_local_rank(gmc), get_global_rank() // 2)
