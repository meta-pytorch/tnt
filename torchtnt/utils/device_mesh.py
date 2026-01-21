# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torchtnt.utils.distributed import get_world_size


class GlobalMeshCoordinator:
    def __init__(
        self,
        dp_shard: int = -1,
        dp_replicate: int = 1,
        tp: Optional[int] = None,
        device_type: str = "cuda",
    ) -> None:
        """
        Initializes the GlobalMeshCoordinator with the specified parameters. This is used to coordinate 1D (fsdp2) and 2D (tp + dp/fsdp2/hsdp) mesh
        for advanced distributed model training / inference.

        Args:
            dp_shard (int): Number of shards for data parallelism. Default is -1, which means infer based on world size.
            dp_replicate (int): Number of replicas for data parallelism. Default is 1.
            tp (Optional[int]): Number of tensor parallelism dimensions. Default is None, which means no tensor parallelism used.
                If wanting to use tensor parallelism, we recommend setting this to 8 to keep TP within intra-node.
            device_type (str): Device type to use. Default is "cuda".

        Example:

            +---------------------------------------------------------+
            |                        replica 0                        |
            | host 0 : |r00|r01|r02|r03|r04|r05|r06|r07|  <-- TP -->  |
            |            ↕   ↕   ↕   ↕   ↕   ↕   ↕   ↕       FSDP     |
            | host 1 : |r08|r09|r10|r11|r12|r13|r14|r15|  <-- TP -->  |
            +---------------------------------------------------------+
            |                        replica 1                        |
            | host 2 : |r16|r17|r18|r19|r20|r21|r22|r23|  <-- TP -->  |
            |            ↕   ↕   ↕   ↕   ↕   ↕   ↕   ↕       FSDP     |
            | host 3 : |r24|r25|r26|r27|r28|r29|r30|r31|  <-- TP -->  |
            +---------------------------------------------------------+

            Legend
            ------
            world_size   : 32
            dp_replicate : 2
            dp_shard     : 2
            tp           : 8
        """

        self.device_mesh: DeviceMesh = create_device_mesh(
            dp_shard, dp_replicate, tp, device_type
        )

        self._dp_replicate_enabled: bool = dp_replicate > 1
        self._tp_enabled: bool = tp is not None

    @property
    def dp_mesh(self) -> DeviceMesh:
        """
        Returns the data parallel mesh (includes replicate and shard dimensions).
        Mesh is directly useable by fsdp2 APIs (fully_shard).
        """
        if self._dp_replicate_enabled:
            return self.device_mesh["dp"]
        return self.device_mesh["dp_shard"]

    @property
    def tp_mesh(self) -> Optional[DeviceMesh]:
        """
        Returns the tensor parallel mesh usable by TP APIs (parallelize_module).
        """
        if self._tp_enabled:
            return self.device_mesh["tp"]

        return None


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
    device_type: str = "cuda",
) -> DeviceMesh:
    """
    Create a DeviceMesh object for the current process group.

    Args:
        dp_shard (int): number of shards for data parallelism. Default is -1, which means we infer the number of shards from the world size.
        dp_replicate (int): number of replicas for data parallelism. Default is 1.
        tp (Optional[int]): number of tensor parallelism dims. Default is None, which means we don't use tensor parallelism.
            If wanting to use tensor parallelism, we recommend setting this to 8 to keep TP within intra-node.
        device_type (str): device type to use. Default is "cuda".

    Returns:
        DeviceMesh: a DeviceMesh object for the current process group

    Note: The returned DeviceMesh will have "dp" and "tp" as the mesh_dim_names. This allows device_mesh["dp"] to be directly used with the
        fsdp2 API, and device_mesh["tp"] to be directly used with the tp API.

    Note: init_process_group should be called prior to this function
    """

    world_size = get_world_size()

    if dp_shard == -1:
        # infer number of dp shards from world size and replicas/tp
        dp_shard = (
            world_size // (dp_replicate)
            if tp is None
            else world_size // (dp_replicate * tp)
        )

    if world_size != dp_shard * dp_replicate * (tp or 1):
        raise ValueError(
            f"World size {world_size} must be divisible by dp_shard={dp_shard} * dp_replicate={dp_replicate} * tp={tp}"
        )

    dims = [dp_replicate, dp_shard] + ([tp] if tp is not None else [])
    names = ["dp_replicate", "dp_shard"] + (["tp"] if tp is not None else [])

    mesh = init_device_mesh(
        device_type=device_type, mesh_shape=tuple(dims), mesh_dim_names=tuple(names)
    )

    # setup submesh for data parallel dimensions
    mesh[("dp_replicate", "dp_shard")]._flatten(mesh_dim_name="dp")

    return mesh
