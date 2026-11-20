# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torchtnt.utils.distributed import get_world_size


@dataclass
class GlobalMeshCoordinator:
    """Coordinates 1D / 2D / 3D device meshes for FSDP2, TP, and EP, including
    shared-axis composition where Tensor Parallelism nests inside Expert
    Parallelism groups for MoE models.

    Legal mesh regimes:
      - ``(tp=1, ep=1)``: FSDP-only.
      - ``(tp>1, ep=1)``: TP-only (dense models, e.g. Qwen3.5-9B).
      - ``(tp=1, ep>1)``: legacy EP-only; permitted (wasteful because
        attention is replicated across EP-group ranks).
      - ``(tp>1, ep>1)``: shared-axis EP+TP (MoE); requires
        ``ep >= tp`` and ``ep % tp == 0`` (TP sub-groups must evenly
        divide each EP group). Violations raise ``ValueError``.

    Context parallelism (``cp > 1``) splits each sequence across ``cp`` ranks
    that share one batch. The dense rank order is ``(dp_replicate, dp_shard,
    cp, tp)``, TP innermost. CP gives three roles three different meshes:

      - ``dp_mesh`` is the data mesh, the ranks that receive distinct
        batches. It excludes CP peers.
      - ``loss_mesh`` (data x cp) is where per-token loss and count sums
        reduce, since a batch's tokens are split across CP peers.
      - ``dense_fsdp_mesh`` and ``fsdp_mesh`` shard parameters over
        ``dp_shard x cp``.

    With ``cp == 1`` every mesh is the one built without CP.

    ``dp_shard = -1`` auto-infers from the world size and the other axes.

    ``tp_mesh``, ``ep_mesh``, ``cp_mesh`` and ``expert_fsdp_mesh`` return
    ``None`` when their axis is inactive.

    Args:
        dp_shard: number of FSDP shards. ``-1`` (default) auto-infers.
        dp_replicate: number of FSDP replicas (HSDP outer). Default ``1``.
        tp: tensor-parallel degree. Default ``1``. Accepts ``None`` for
            back-compat; normalized to ``1`` internally.
        ep: expert-parallel degree. Default ``1``.
        device_type: device type. Default ``"cuda"``.
        cp: context-parallel degree. Default ``1``.
    """

    dp_shard: int = -1
    dp_replicate: int = 1
    tp: Optional[int] = None
    ep: int = 1
    device_type: str = "cuda"
    cp: int = 1

    # Set in __post_init__; declared here so dataclass field-discovery
    # surfaces them and type-checkers see them as instance attributes.
    device_mesh: DeviceMesh = field(init=False)
    _dp_replicate_enabled: bool = field(init=False)
    _tp_enabled: bool = field(init=False)
    _ep_enabled: bool = field(init=False)
    _cp_enabled: bool = field(init=False)

    def __post_init__(self) -> None:
        # Backwards-compat: ``tp=None`` (legacy callers) is normalized to
        # 1. Internally we always reason in terms of an int >= 1.
        if self.tp is None:
            self.tp = 1
        if not isinstance(self.tp, int) or self.tp < 1:
            raise ValueError(f"tp must be int >= 1, got {self.tp!r}")
        if not isinstance(self.ep, int) or self.ep < 1:
            raise ValueError(f"ep must be int >= 1, got {self.ep!r}")
        if not isinstance(self.cp, int) or self.cp < 1:
            raise ValueError(f"cp must be int >= 1, got {self.cp!r}")

        assert isinstance(self.tp, int)
        tp: int = self.tp

        # (tp=1, ep>1) — no extra checks; legacy EP-only is permitted.
        if self.ep > 1 and tp > 1:
            if tp > self.ep:
                raise ValueError(
                    f"tp ({tp}) must be <= ep ({self.ep}) under "
                    "shared-axis EP+TP. EP is the outer axis; TP nests "
                    "inside EP. To use more TP, raise ep first."
                )
            if self.ep % tp != 0:
                raise ValueError(
                    f"ep ({self.ep}) must be divisible by tp ({tp}). "
                    "TP sub-groups must EVENLY divide each EP group."
                )

        self.device_mesh = create_device_mesh(
            dp_shard=self.dp_shard,
            dp_replicate=self.dp_replicate,
            tp=tp,
            ep=self.ep,
            device_type=self.device_type,
            cp=self.cp,
        )
        self._dp_replicate_enabled = self.dp_replicate > 1
        self._tp_enabled = tp > 1
        self._ep_enabled = self.ep > 1
        self._cp_enabled = self.cp > 1

    @property
    def dp_mesh(self) -> DeviceMesh:
        """Data mesh: the ranks that receive distinct batches.

        Samplers and dataloaders shard over it. With ``cp == 1`` it is
        :attr:`dense_fsdp_mesh`. With ``cp > 1`` it excludes CP peers, which
        share a batch, so FSDP wrapping and per-token loss sums must not use
        it: they belong on :attr:`fsdp_mesh` / :attr:`dense_fsdp_mesh` and
        :attr:`loss_mesh`, or CP peers' gradients and tokens are left out.
        """
        if not self._cp_enabled:
            return self.dense_fsdp_mesh
        # pyrefly: ignore [missing-attribute]
        return self.device_mesh._dp_data_mesh

    @property
    def loss_mesh(self) -> DeviceMesh:
        """Ranks whose tokens partition the step's batches: data x cp.

        Reduce per-token loss, token-count and metric sums over this mesh;
        :attr:`dp_mesh` leaves out the tokens held by CP peers. Data x cp is
        every rank that shares a TP index, HSDP replicas included, which is
        exactly :attr:`dense_fsdp_mesh` in every regime; so this returns it,
        and its size is also the width FSDP averages dense gradients over.
        """
        return self.dense_fsdp_mesh

    @property
    def cp_mesh(self) -> Optional[DeviceMesh]:
        """CP grid of size ``cp``: the ranks that split one batch's sequences.

        Returns ``None`` when ``cp == 1``.
        """
        if not self._cp_enabled:
            return None
        # pyrefly: ignore [missing-attribute]
        return self.device_mesh._cp_view["cp"]

    @property
    def dense_fsdp_mesh(self) -> DeviceMesh:
        """FSDP grid for DENSE (non-expert) parameters.

        Width: ``world / tp`` in every regime, i.e. every rank that shares a
        TP index (``dp_replicate * dp_shard * cp`` when ``ep == 1``). CP peers
        shard parameters too, so they are inside this grid.

        Sub-axis of ``dense_mesh`` so it shares a parent with
        :attr:`tp_mesh` — FSDP2 can promote dense DTensors over the
        ``(fsdp, tp)`` 2-D sub-mesh.

        Returns the flattened ``dp_dense`` axis when ``dp_replicate > 1``
        so HSDP-style callers see a single composite data-parallel mesh.
        """
        if self._dp_replicate_enabled:
            # pyre-ignore[16]
            return self.device_mesh._dp_dense_mesh

        # pyre-ignore[16]
        return self.device_mesh._dense_view["fsdp"]

    @property
    def fsdp_mesh(self) -> DeviceMesh:
        """FSDP2 wrapping mesh for DENSE params, preserving the HSDP 2-D grid.

        Unlike :attr:`dense_fsdp_mesh` — which FLATTENS ``(dp_replicate, fsdp)``
        into one 1-D axis so DP-group / dataloader consumers see a single
        composite data-parallel mesh — this returns the UN-flattened 2-D
        ``(dp_replicate, fsdp)`` mesh when ``dp_replicate > 1``. Passing that
        2-D mesh to ``fully_shard`` yields true HSDP: parameters are sharded
        within each replica group (the ``fsdp`` axis) and replicated across
        groups (the ``dp_replicate`` axis).

        Falls back to the 1-D ``fsdp`` sub-mesh when ``dp_replicate == 1``
        (no replication -> plain FSDP ``FULL_SHARD``), so callers that do not
        opt into HSDP see identical behavior to :attr:`dense_fsdp_mesh`.

        With ``cp > 1`` the ``fsdp`` axis spans ``dp_shard x cp``.
        """
        if self._dp_replicate_enabled:
            # pyre-ignore[16]
            return self.device_mesh._dense_view["dp_replicate", "fsdp"]
        # pyre-ignore[16]
        return self.device_mesh._dense_view["fsdp"]

    @property
    def tp_mesh(self) -> Optional[DeviceMesh]:
        """TP grid of size ``tp``. Returns ``None`` when ``tp == 1``.

        Shared-axis (``ep > 1``): the intra-EP-group sub-mesh.
        TP-only mode (``ep == 1``): the standard TP mesh.

        Sub-axis of ``dense_mesh`` (shared parent with
        :attr:`dense_fsdp_mesh`).
        """
        if not self._tp_enabled:
            return None
        # pyre-ignore[16]
        return self.device_mesh._dense_view["tp"]

    @property
    def ep_mesh(self) -> Optional[DeviceMesh]:
        """EP grid of size ``ep``. Returns ``None`` when ``ep == 1``.

        Used by ``apply_expert_parallel`` as the AllToAll group.

        Sub-axis of ``sparse_mesh`` (shared parent with
        :attr:`expert_fsdp_mesh`).
        """
        if not self._ep_enabled:
            return None
        # pyre-ignore[16]
        return self.device_mesh._sparse_view["ep"]

    @property
    def expert_fsdp_mesh(self) -> Optional[DeviceMesh]:
        """
        FSDP grid for EXPERT parameters. Returns ``None`` when ``ep == 1``.

        Width: ``world / ep``. NUMERICALLY DIFFERENT from
        :attr:`dense_fsdp_mesh` when ``tp < ep``.

        Sub-axis of ``sparse_mesh`` (shared parent with :attr:`ep_mesh`)
        so FSDP2 can promote expert DTensors over the ``(efsdp, ep)``
        2-D sub-mesh.

        Numerical relationship to :attr:`dense_fsdp_mesh` under
        shared-axis:

          - ``tp == ep``:
                ``expert_fsdp_mesh.size() == dense_fsdp_mesh.size()``.
                Physical rank groups happen to be equivalent groupings.
          - ``tp <  ep``:
                ``expert_fsdp_mesh.size() <  dense_fsdp_mesh.size()``.
                Different sizes AND different physical groupings.
          - ``tp >  ep``: rejected by ``__post_init__`` validator.
        """
        if not self._ep_enabled:
            return None
        if self._dp_replicate_enabled:
            # pyre-ignore[16]
            return self.device_mesh._dp_sparse_mesh
        # pyre-ignore[16]
        return self.device_mesh._sparse_view["efsdp"]


def get_dp_mesh(global_mesh: GlobalMeshCoordinator) -> DeviceMesh:
    """
    Retrieves the data parallel mesh from the global mesh coordinator.

    Args:
        global_mesh (GlobalMeshCoordinator): The global mesh coordinator instance.

    Returns:
        DeviceMesh: The data parallel mesh.
    """
    return global_mesh.dp_mesh


def get_dp_mesh_size(global_mesh: GlobalMeshCoordinator) -> int:
    """
    Retrieves the size of the data parallel mesh from the global mesh coordinator.

    Args:
        global_mesh (GlobalMeshCoordinator): The global mesh coordinator instance.

    Returns:
        int: The size of the data parallel mesh.
    """
    return global_mesh.dp_mesh.size()


def get_dp_local_rank(global_mesh: GlobalMeshCoordinator) -> int:
    """
    Retrieves the local rank within the data parallel mesh from the global mesh coordinator.

    Args:
        global_mesh (GlobalMeshCoordinator): The global mesh coordinator instance.

    Returns:
        int: The local rank within the data parallel mesh.
    """
    return global_mesh.dp_mesh.get_local_rank()


def create_device_mesh(
    dp_shard: int = -1,
    dp_replicate: int = 1,
    tp: Optional[int] = None,
    ep: int = 1,
    device_type: str = "cuda",
    cp: int = 1,
) -> DeviceMesh:
    """Build a flat 1D world mesh, then ``_unflatten`` it into TWO
    independent 3-D named parent views: ``dense_mesh`` and ``sparse_mesh``.

    The returned ``DeviceMesh`` has private attributes ``_dense_view``
    and ``_sparse_view`` that hold the two named 3-D parent views;
    ``_sparse_view`` is ``None`` when ``ep == 1``. With ``cp > 1`` the dense
    view's ``fsdp`` axis spans ``dp_shard x cp``, ``_cp_view`` splits it into
    ``(dp_replicate, dp_shard, cp, tp)`` and ``_dp_data_mesh`` is the data
    mesh; both are ``None`` when ``cp == 1``.

    Args:
        dp_shard: FSDP shard width. ``-1`` (default) auto-infers from the
            world size and the other axes (using ``ep`` as the divisor
            when ``ep > 1``, else ``cp * tp``).
        dp_replicate: HSDP outer replicate width. Default ``1``.
        tp: tensor-parallel degree. ``None`` is treated as ``1`` for
            backwards compatibility with callers that have not adopted
            the new spec.
        ep: expert-parallel degree. Default ``1``.
        device_type: device type. Default ``"cuda"``.
        cp: context-parallel degree. Default ``1``.

    Returns:
        ``DeviceMesh`` with ``_dense_view`` (always set), ``_sparse_view``
        (set only when ``ep > 1``), ``_cp_view`` and ``_dp_data_mesh`` (set
        only when ``cp > 1``) attributes attached.

    Note: ``init_process_group`` should be called prior to this function.
    """
    # Normalize legacy ``tp=None`` callers to ``tp=1`` so the conservation
    # law and the unflatten degrees stay int-typed throughout.
    tp_int = 1 if tp is None else int(tp)

    world_size = get_world_size()

    # Conservation law: differs between TP-only mode and shared-axis.
    if ep > 1:
        # Shared-axis: world = dp_replicate * dp_shard * ep
        # (TP is folded into each EP group; consumes no fresh ranks).
        if dp_shard == -1:
            dp_shard = world_size // (dp_replicate * ep)
        if world_size != dp_shard * dp_replicate * ep:
            raise ValueError(
                f"World size {world_size} != dp_shard={dp_shard} * "
                f"dp_replicate={dp_replicate} * ep={ep}. "
                "Under shared-axis EP+TP (ep>1), tp is NOT in the product."
            )
    else:
        # TP-only mode / pure FSDP: world = dp_replicate * dp_shard * cp * tp.
        if dp_shard == -1:
            dp_shard = world_size // (dp_replicate * cp * tp_int)
        if world_size != dp_shard * dp_replicate * cp * tp_int:
            raise ValueError(
                f"World size {world_size} != dp_shard={dp_shard} * "
                f"dp_replicate={dp_replicate} * cp={cp} * tp={tp_int}. "
                "Under TP-only mode (ep=1), ep is NOT in the product."
            )

    # 1) Build the flat world mesh.
    world_mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(world_size,),
        mesh_dim_names=("world",),
    )

    if ep > 1:
        # === Shared-axis (ep > 1) ===
        #   fsdp = world / (dp_replicate * tp)
        fsdp = world_size // (dp_replicate * tp_int)
        if fsdp % cp != 0:
            raise ValueError(
                f"Dense FSDP width {fsdp} (world / (dp_replicate * tp)) is not "
                f"divisible by cp={cp}; CP peers must split it evenly."
            )
        dense_mesh = world_mesh._unflatten(
            0,
            (dp_replicate, fsdp, tp_int),
            ("dp_replicate", "fsdp", "tp"),
        )

        # Sparse view (3-D): (dp_replicate, efsdp, ep).
        #   efsdp = fsdp * tp / ep = dp_shard * tp / ep
        efsdp = (fsdp * tp_int) // ep
        sparse_mesh = world_mesh._unflatten(
            0,
            (dp_replicate, efsdp, ep),
            ("dp_replicate", "efsdp", "ep"),
        )

        # Flatten dp_replicate * fsdp (dense) for HSDP-style callers.
        dp_dense_mesh = dense_mesh[("dp_replicate", "fsdp")]._flatten(
            mesh_dim_name="dp_dense"
        )
        # Flatten dp_replicate * efsdp (sparse) — the data-parallel scope
        # for expert FSDP under HSDP.
        dp_sparse_mesh = sparse_mesh[("dp_replicate", "efsdp")]._flatten(
            mesh_dim_name="dp_sparse"
        )

        world_mesh._dense_view = dense_mesh  # pyre-ignore[16]
        world_mesh._sparse_view = sparse_mesh  # pyre-ignore[16]
        world_mesh._dp_dense_mesh = dp_dense_mesh  # pyre-ignore[16]
        world_mesh._dp_sparse_mesh = dp_sparse_mesh  # pyre-ignore[16]
    else:
        # === TP-only mode / pure FSDP (ep == 1) ===
        fsdp = dp_shard * cp
        dense_mesh = world_mesh._unflatten(
            0,
            (dp_replicate, fsdp, tp_int),
            ("dp_replicate", "fsdp", "tp"),
        )
        dp_dense_mesh = dense_mesh[("dp_replicate", "fsdp")]._flatten(
            mesh_dim_name="dp_dense"
        )
        world_mesh._dense_view = dense_mesh  # pyre-ignore[16]
        world_mesh._sparse_view = None  # pyre-ignore[16]
        world_mesh._dp_dense_mesh = dp_dense_mesh  # pyre-ignore[16]
        world_mesh._dp_sparse_mesh = None  # pyre-ignore[16]

    world_mesh._cp_view = None  # pyrefly: ignore [missing-attribute]
    world_mesh._dp_data_mesh = None  # pyrefly: ignore [missing-attribute]
    if cp > 1:
        # Split the dense FSDP axis rather than unflattening the world again:
        # this creates only the dp_shard and cp groups. CP sits inside dp_shard
        # and outside tp, so CP peers are adjacent ranks and the data mesh is
        # the FSDP grid without them.
        cp_view = dense_mesh._unflatten("fsdp", (fsdp // cp, cp), ("dp_shard", "cp"))
        if dp_replicate > 1:
            dp_data_mesh = cp_view[("dp_replicate", "dp_shard")]._flatten(
                mesh_dim_name="dp_data"
            )
        else:
            dp_data_mesh = cp_view["dp_shard"]
        world_mesh._cp_view = cp_view  # pyrefly: ignore [missing-attribute]
        world_mesh._dp_data_mesh = dp_data_mesh  # pyrefly: ignore [missing-attribute]

    return world_mesh
