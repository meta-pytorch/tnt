# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

        with tc.assertRaisesRegex(ValueError, "World size 4 != "):
            create_device_mesh(dp_shard=-1, dp_replicate=1, tp=8, device_type="cpu")

        with tc.assertRaisesRegex(ValueError, "World size 4 != "):
            create_device_mesh(dp_shard=-1, dp_replicate=1, tp=3, device_type="cpu")

        device_mesh = create_device_mesh(
            dp_shard=-1, dp_replicate=2, tp=None, device_type="cpu"
        )

        # The new API exposes the dense view (3-D: dp_replicate, fsdp, tp);
        # ``fsdp`` is the renamed shard axis. With dp_replicate=2 and tp=1
        # (None normalized) on world=4, fsdp must be 2.
        # pyre-ignore[16]
        tc.assertEqual(device_mesh._dense_view["fsdp"].size(), 2)


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

        # Under the new 3-regime contract, ``tp=1`` is regime 1 (FSDP-only)
        # and ``_tp_enabled`` is False. Only ``tp > 1`` enables TP.
        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=1, device_type="cpu"
        )
        tc.assertFalse(gmc._dp_replicate_enabled)
        tc.assertFalse(gmc._tp_enabled)

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

        # tp=None, dp_replicate=1 → dp_mesh is dense_fsdp_mesh, size = world = 4.
        # NOTE: DeviceMesh sub-mesh objects from ``_create_sub_mesh`` are
        # fresh instances per call; compare by observable contract
        # (size, mesh_dim_names, root identity) instead of object equality.
        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=None, device_type="cpu"
        )
        tc.assertEqual(gmc.dp_mesh.mesh_dim_names, ("fsdp",))
        tc.assertEqual(gmc.dp_mesh.size(), 4)
        tc.assertEqual(get_dp_mesh_size(gmc), 4)
        tc.assertEqual(get_dp_local_rank(gmc), get_global_rank())

        # tp=None, dp_replicate=2 → HSDP path: dp_mesh is the flattened
        # dp_dense composite axis (dp_replicate * fsdp), size 4.
        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=2, tp=None, device_type="cpu"
        )
        tc.assertEqual(gmc.dp_mesh.mesh_dim_names, ("dp_dense",))
        tc.assertEqual(gmc.dp_mesh.size(), 4)
        tc.assertEqual(get_dp_mesh_size(gmc), 4)
        tc.assertEqual(get_dp_local_rank(gmc), get_global_rank())

        # tp=2 → dp_mesh is dense_view["fsdp"], size = world / tp = 2.
        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=2, device_type="cpu"
        )
        tc.assertEqual(gmc.dp_mesh.mesh_dim_names, ("fsdp",))
        tc.assertEqual(gmc.dp_mesh.size(), 2)
        tc.assertEqual(get_dp_mesh_size(gmc), 2)
        tc.assertEqual(get_dp_local_rank(gmc), get_global_rank() // 2)


class TestSharedAxisRegimeValidator(unittest.TestCase):
    """3-regime validator coverage (SA-A: plan §2.1)."""

    def test_three_regime_validation_tp_only_accepted(self) -> None:
        spawn_multi_process(
            4, "gloo", self._test_three_regime_validation_tp_only_accepted
        )

    @staticmethod
    def _test_three_regime_validation_tp_only_accepted() -> None:
        # Regime 2: (tp>1, ep=1) — TP-only mode (dense models).
        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=2, ep=1, device_type="cpu"
        )
        tc = unittest.TestCase()
        tc.assertTrue(gmc._tp_enabled)
        tc.assertFalse(gmc._ep_enabled)

    def test_three_regime_validation_shared_axis_accepted(self) -> None:
        spawn_multi_process(
            4, "gloo", self._test_three_regime_validation_shared_axis_accepted
        )

    @staticmethod
    def _test_three_regime_validation_shared_axis_accepted() -> None:
        # Regime 3: (tp>1, ep>1) with ep>=tp and ep%tp==0 — shared-axis.
        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=2, ep=4, device_type="cpu"
        )
        tc = unittest.TestCase()
        tc.assertTrue(gmc._tp_enabled)
        tc.assertTrue(gmc._ep_enabled)

    def test_three_regime_validation_tp_eq_ep_accepted(self) -> None:
        spawn_multi_process(
            4, "gloo", self._test_three_regime_validation_tp_eq_ep_accepted
        )

    @staticmethod
    def _test_three_regime_validation_tp_eq_ep_accepted() -> None:
        # Regime 3 edge case: tp == ep.
        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=4, ep=4, device_type="cpu"
        )
        tc = unittest.TestCase()
        tc.assertTrue(gmc._tp_enabled)
        tc.assertTrue(gmc._ep_enabled)

    def test_three_regime_rejects_tp_gt_ep(self) -> None:
        spawn_multi_process(1, "gloo", self._test_three_regime_rejects_tp_gt_ep)

    @staticmethod
    def _test_three_regime_rejects_tp_gt_ep() -> None:
        tc = unittest.TestCase()
        # (tp>ep, ep>1) — TP cannot be wider than EP.
        with tc.assertRaisesRegex(ValueError, r"tp \(4\) must be <= ep \(2\)"):
            GlobalMeshCoordinator(
                dp_shard=-1, dp_replicate=1, tp=4, ep=2, device_type="cpu"
            )

    def test_three_regime_rejects_ep_not_divisible_by_tp(self) -> None:
        spawn_multi_process(
            1, "gloo", self._test_three_regime_rejects_ep_not_divisible_by_tp
        )

    @staticmethod
    def _test_three_regime_rejects_ep_not_divisible_by_tp() -> None:
        tc = unittest.TestCase()
        # (ep%tp != 0, ep>1) — TP sub-groups must EVENLY divide each EP group.
        with tc.assertRaisesRegex(
            ValueError, r"ep \(4\) must be divisible by tp \(3\)"
        ):
            GlobalMeshCoordinator(
                dp_shard=-1, dp_replicate=1, tp=3, ep=4, device_type="cpu"
            )

    def test_world_size_law_shared_axis(self) -> None:
        spawn_multi_process(8, "gloo", self._test_world_size_law_shared_axis)

    @staticmethod
    def _test_world_size_law_shared_axis() -> None:
        # Shared-axis conservation: world = dp_replicate * dp_shard * ep.
        # At world=8 with (tp=2, ep=4), dp_shard=-1 must infer to
        # 8 / (1 * 4) = 2.
        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=2, ep=4, device_type="cpu"
        )
        tc = unittest.TestCase()
        # Sparse view shape: (dp_replicate=1, efsdp, ep=4).
        # efsdp = dp_shard * tp / ep = 2 * 2 / 4 = 1.
        # ... but numerically dp_shard=2 here, so efsdp = 2 * 2 / 4 = 1
        # only when dp_shard=1. With auto-inferred dp_shard=2, efsdp =
        # world / (dp_replicate * ep) = 8 / 4 = 2. The expert_fsdp size
        # is the latter (the sparse-view formula).
        expert_fsdp_mesh = gmc.expert_fsdp_mesh
        ep_mesh = gmc.ep_mesh
        tp_mesh = gmc.tp_mesh
        assert expert_fsdp_mesh is not None
        assert ep_mesh is not None
        assert tp_mesh is not None
        tc.assertEqual(expert_fsdp_mesh.size(), 2)
        tc.assertEqual(ep_mesh.size(), 4)
        # Dense view shape: (1, fsdp, tp=2). fsdp = world/tp = 4.
        tc.assertEqual(gmc.dense_fsdp_mesh.size(), 4)
        tc.assertEqual(tp_mesh.size(), 2)

    def test_world_size_law_tp_only(self) -> None:
        spawn_multi_process(8, "gloo", self._test_world_size_law_tp_only)

    @staticmethod
    def _test_world_size_law_tp_only() -> None:
        # TP-only conservation: world = dp_replicate * dp_shard * tp.
        # At world=8 with (tp=4, ep=1), dp_shard=-1 must infer to
        # 8 / (1 * 4) = 2.
        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=4, ep=1, device_type="cpu"
        )
        tc = unittest.TestCase()
        tp_mesh = gmc.tp_mesh
        assert tp_mesh is not None
        tc.assertEqual(tp_mesh.size(), 4)
        tc.assertEqual(gmc.dense_fsdp_mesh.size(), 2)


class TestSharedAxisMeshDerivation(unittest.TestCase):
    """Sub-mesh derivation tests (SA-A: plan §2.1, dense_fsdp ≠ expert_fsdp)."""

    def test_dense_fsdp_size_when_tp_lt_ep(self) -> None:
        spawn_multi_process(8, "gloo", self._test_dense_fsdp_size_when_tp_lt_ep)

    @staticmethod
    def _test_dense_fsdp_size_when_tp_lt_ep() -> None:
        # (tp=2, ep=4) at world=8:
        #   dense_fsdp width = world / tp = 4
        #   expert_fsdp width = world / ep = 2
        # Different sizes — the v3 contract that incorrectly equated them
        # is regression-gated here.
        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=2, ep=4, device_type="cpu"
        )
        tc = unittest.TestCase()
        expert_fsdp_mesh = gmc.expert_fsdp_mesh
        assert expert_fsdp_mesh is not None
        tc.assertEqual(gmc.dense_fsdp_mesh.size(), 4)
        tc.assertEqual(expert_fsdp_mesh.size(), 2)
        tc.assertNotEqual(gmc.dense_fsdp_mesh.size(), expert_fsdp_mesh.size())

    def test_dense_fsdp_eq_expert_fsdp_when_tp_eq_ep(self) -> None:
        spawn_multi_process(
            8, "gloo", self._test_dense_fsdp_eq_expert_fsdp_when_tp_eq_ep
        )

    @staticmethod
    def _test_dense_fsdp_eq_expert_fsdp_when_tp_eq_ep() -> None:
        # (tp=4, ep=4) at world=8: tp == ep, so dense_fsdp and expert_fsdp
        # have the same size (= world / ep = 2).
        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=4, ep=4, device_type="cpu"
        )
        tc = unittest.TestCase()
        expert_fsdp_mesh = gmc.expert_fsdp_mesh
        assert expert_fsdp_mesh is not None
        tc.assertEqual(gmc.dense_fsdp_mesh.size(), 2)
        tc.assertEqual(expert_fsdp_mesh.size(), 2)

    def test_expert_fsdp_mesh_is_subaxis_of_sparse_view(self) -> None:
        spawn_multi_process(
            8, "gloo", self._test_expert_fsdp_mesh_is_subaxis_of_sparse_view
        )

    @staticmethod
    def _test_expert_fsdp_mesh_is_subaxis_of_sparse_view() -> None:
        # FSDP2-compatibility gate for the SPARSE pair: expert_fsdp_mesh
        # and ep_mesh must share a parent (both unflattened from the same
        # _sparse_view). DeviceMesh sub-mesh accessors return fresh
        # objects per call, so check the contract via the root mesh
        # (identity-stable) and the mesh_dim_names.
        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=2, ep=4, device_type="cpu"
        )
        tc = unittest.TestCase()
        sparse_view = getattr(gmc.device_mesh, "_sparse_view", None)
        tc.assertIsNotNone(sparse_view)
        # Both sub-meshes are derived from the SAME sparse_view object
        # (held identity-stably on world_mesh._sparse_view).
        ep_mesh = gmc.ep_mesh
        expert_fsdp_mesh = gmc.expert_fsdp_mesh
        assert ep_mesh is not None
        assert expert_fsdp_mesh is not None
        tc.assertIs(ep_mesh._get_root_mesh(), expert_fsdp_mesh._get_root_mesh())
        # And the named axes match what the sparse-view contract promises.
        tc.assertEqual(ep_mesh.mesh_dim_names, ("ep",))
        tc.assertEqual(expert_fsdp_mesh.mesh_dim_names, ("efsdp",))

    def test_dense_fsdp_mesh_is_subaxis_of_dense_view(self) -> None:
        spawn_multi_process(
            8, "gloo", self._test_dense_fsdp_mesh_is_subaxis_of_dense_view
        )

    @staticmethod
    def _test_dense_fsdp_mesh_is_subaxis_of_dense_view() -> None:
        # FSDP2-compatibility gate for the DENSE pair: dense_fsdp_mesh and
        # tp_mesh must share a parent (both unflattened from the same
        # _dense_view). DeviceMesh sub-mesh accessors return fresh objects
        # per call, so check the contract via the root mesh
        # (identity-stable) and the mesh_dim_names.
        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=2, ep=4, device_type="cpu"
        )
        tc = unittest.TestCase()
        dense_view = getattr(gmc.device_mesh, "_dense_view", None)
        tc.assertIsNotNone(dense_view)
        tp_mesh = gmc.tp_mesh
        assert tp_mesh is not None
        tc.assertIs(tp_mesh._get_root_mesh(), gmc.dense_fsdp_mesh._get_root_mesh())
        tc.assertEqual(tp_mesh.mesh_dim_names, ("tp",))
        tc.assertEqual(gmc.dense_fsdp_mesh.mesh_dim_names, ("fsdp",))

    def test_world_mesh_independent_views(self) -> None:
        spawn_multi_process(8, "gloo", self._test_world_mesh_independent_views)

    @staticmethod
    def _test_world_mesh_independent_views() -> None:
        # Both views cover the SAME world ranks via different
        # decompositions. world_size = dense_fsdp * tp = expert_fsdp * ep.
        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=2, ep=4, device_type="cpu"
        )
        tc = unittest.TestCase()
        world_size = 8
        # Dense view: (1, 4, 2) → 1 * 4 * 2 = 8
        tp_mesh = gmc.tp_mesh
        ep_mesh = gmc.ep_mesh
        expert_fsdp_mesh = gmc.expert_fsdp_mesh
        assert tp_mesh is not None
        assert ep_mesh is not None
        assert expert_fsdp_mesh is not None
        tc.assertEqual(gmc.dense_fsdp_mesh.size() * tp_mesh.size(), world_size)
        # Sparse view: (1, 2, 4) → 1 * 2 * 4 = 8
        tc.assertEqual(expert_fsdp_mesh.size() * ep_mesh.size(), world_size)
        # Independent objects (two distinct _unflatten calls).
        tc.assertIsNot(
            getattr(gmc.device_mesh, "_dense_view", None),
            getattr(gmc.device_mesh, "_sparse_view", None),
        )

    def test_dp_mesh_alias_returns_dense_fsdp(self) -> None:
        spawn_multi_process(4, "gloo", self._test_dp_mesh_alias_returns_dense_fsdp)

    @staticmethod
    def _test_dp_mesh_alias_returns_dense_fsdp() -> None:
        # Backwards-compat alias contract: dp_mesh delegates to
        # dense_fsdp_mesh (same observable contract).
        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=2, device_type="cpu"
        )
        tc = unittest.TestCase()
        # The two properties produce sub-meshes from the same _dense_view
        # with the same axis name; DeviceMesh creates fresh sub-mesh
        # objects per access, so compare by observable contract.
        tc.assertEqual(gmc.dp_mesh.mesh_dim_names, gmc.dense_fsdp_mesh.mesh_dim_names)
        tc.assertEqual(gmc.dp_mesh.size(), gmc.dense_fsdp_mesh.size())
        tc.assertIs(gmc.dp_mesh._get_root_mesh(), gmc.dense_fsdp_mesh._get_root_mesh())


class TestEPDeviceMeshTpOnly(unittest.TestCase):
    """Backwards-compat: ep=1 (default) regimes 1 and 2 still work."""

    def test_backward_compat_no_ep_4_ranks(self) -> None:
        """ep=1 (default) FSDP-only path still produces a valid mesh."""
        spawn_multi_process(4, "gloo", self._test_backward_compat_no_ep_4_ranks)

    @staticmethod
    def _test_backward_compat_no_ep_4_ranks() -> None:
        tc = unittest.TestCase()
        gmc_today = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=None, device_type="cpu"
        )
        gmc_new = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=None, ep=1, device_type="cpu"
        )
        # dp_mesh is the dense_fsdp_mesh for both; sizes match.
        tc.assertEqual(gmc_new.dp_mesh.size(), 4)
        tc.assertEqual(gmc_today.dp_mesh.size(), 4)
        tc.assertIsNone(gmc_new.ep_mesh)
        tc.assertIsNone(gmc_new.expert_fsdp_mesh)
        tc.assertFalse(gmc_new._ep_enabled)
        tc.assertFalse(gmc_today._ep_enabled)


class TestSharedAxisAcceptsLegacyTpNone(unittest.TestCase):
    """Backwards-compat: tp=None (legacy) is normalized to tp=1."""

    def test_tp_none_is_regime_one(self) -> None:
        spawn_multi_process(1, "gloo", self._test_tp_none_is_regime_one)

    @staticmethod
    def _test_tp_none_is_regime_one() -> None:
        tc = unittest.TestCase()
        gmc = GlobalMeshCoordinator(
            dp_shard=-1, dp_replicate=1, tp=None, ep=1, device_type="cpu"
        )
        # Internally tp is normalized to 1.
        tc.assertEqual(gmc.tp, 1)
        tc.assertFalse(gmc._tp_enabled)
        tc.assertIsNone(gmc.tp_mesh)


class TestConservationViolation(unittest.TestCase):
    def test_conservation_violation_raises(self) -> None:
        spawn_multi_process(4, "gloo", self._test_conservation_violation_raises)

    @staticmethod
    def _test_conservation_violation_raises() -> None:
        tc = unittest.TestCase()
        # ep=3 doesn't divide world=4 under shared-axis; raises before
        # the unflatten step.
        with tc.assertRaisesRegex(ValueError, "World size 4 != "):
            create_device_mesh(
                dp_shard=2, dp_replicate=1, tp=2, ep=3, device_type="cpu"
            )
