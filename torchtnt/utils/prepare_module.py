# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import (
    Any,
    Callable,
    Collection,
    ContextManager,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
import torch.distributed as dist
from pyre_extensions import none_throws
from torch.distributed import ProcessGroup
from torch.distributed._composable_state import _get_module_state
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.parallel.style import ParallelStyle
from torchtnt.utils.device_mesh import GlobalMeshCoordinator
from torchtnt.utils.precision import convert_precision_str_to_dtype

try:
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        fully_shard,
        MixedPrecisionPolicy,
    )
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
except ImportError:

    def noop(*args: Any, **kwargs: Any) -> None:
        pass

    class NOOP:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    fully_shard = noop
    MixedPrecisionPolicy = NOOP
    CPUOffloadPolicy = NOOP
    FSDPState = NOOP

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType as _StateDictType,
)
from torch.distributed.fsdp.api import OptimStateDictConfig, StateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch as _BackwardPrefetch,
    CPUOffload,
    MixedPrecision as _MixedPrecision,
    ShardingStrategy as _ShardingStrategy,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtnt.utils.fsdp_utils import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torchtnt.utils.rank_zero_log import rank_zero_info, rank_zero_warn
from torchtnt.utils.version import is_torch_version_geq


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    """Dataclass representing a parallelization strategy"""

    pass


@dataclass
class NOOPStrategy(Strategy):
    """
    Dataclass representing a no-op strategy. Nothing is applied to the module, and no device transfer occurs
    Use this strategy if applying custom wrapping to module prior to passing it into class:`~torchtnt.framework.auto_unit.AutoUnit`
    or into :py:func:`~torchtnt.utils.prepare_module.prepare_module`
    """

    pass


@dataclass
class DDPStrategy(Strategy):
    """
    Dataclass representing the `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ strategy.

    Includes params for registering `DDP communication hooks <https://pytorch.org/docs/stable/ddp_comm_hooks.html>`_ and `syncing batch norm <https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html>`_.
    """

    # DDP Constructor params
    output_device: Optional[Union[int, torch.device]] = None
    dim: int = 0
    broadcast_buffers: bool = True
    process_group: Optional[ProcessGroup] = None
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    check_reduction: bool = False
    gradient_as_bucket_view: bool = False
    static_graph: bool = False

    # DDP Comm Hook params
    comm_state: Optional[object] = None
    comm_hook: Optional[
        Callable[[object, dist.GradBucket], torch.futures.Future[torch.Tensor]]
    ] = None

    # SyncBatchNorm params
    sync_batchnorm: bool = False


@dataclass
class FSDPStrategy(Strategy):
    """Dataclass representing the `FullyShardedDataParallel <https://pytorch.org/docs/stable/fsdp.html>`_ strategy"""

    process_group: Optional[ProcessGroup] = None
    sharding_strategy: Optional[Union[str, _ShardingStrategy]] = None
    cpu_offload: Optional[CPUOffload] = None
    auto_wrap_policy: Optional[Callable[[torch.nn.Module, bool, int], bool]] = None
    backward_prefetch: Optional[Union[str, _BackwardPrefetch]] = (
        _BackwardPrefetch.BACKWARD_PRE
    )
    mixed_precision: Optional[Union[_MixedPrecision, MixedPrecision]] = None
    ignored_modules: Optional[Iterable[torch.nn.Module]] = None
    param_init_fn: Optional[Callable[[torch.nn.Module], None]] = None
    sync_module_states: bool = False
    forward_prefetch: bool = False
    limit_all_gathers: bool = True
    use_orig_params: bool = False

    # FSDP set_state_dict_type params: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type
    # for setting type of state dict for checkpointing
    state_dict_type: Optional[Union[str, _StateDictType]] = None
    state_dict_config: Optional[StateDictConfig] = None
    optim_state_dict_config: Optional[OptimStateDictConfig] = None

    def __post_init__(self) -> None:
        if isinstance(self.sharding_strategy, str):
            self.sharding_strategy = ShardingStrategy.to_native_sharding_strategy(
                self.sharding_strategy
            )

        if isinstance(self.backward_prefetch, str):
            self.backward_prefetch = BackwardPrefetch.to_native_backward_prefetch(
                self.backward_prefetch
            )

        if isinstance(self.state_dict_type, str):
            self.state_dict_type = StateDictType.to_native_state_dict_type(
                self.state_dict_type
            )

        if isinstance(self.mixed_precision, MixedPrecision):
            self.mixed_precision = self.mixed_precision.to_native_mixed_precision()


@dataclass
class FSDP2Strategy(Strategy):
    """
    Dataclass representing the `FSDP2 <https://pytorch.org/docs/2.6/distributed.fsdp.fully_shard.html>`_ strategy.
    For more details on the args, see the link.

    Args:
        modules_to_shard: A list of modules that should be sharded across devices. Options are 'all' to shard all submodules, or a list of module names/module types. Specify None to not shard any modules with this flag.
        shard_predicates: A list of predicates to decide which modules to shard with FSDP. Each predicate takes a module name (fqn) and the module itself. If any predicate returns True, the submodule is sharded.
        reshard_after_forward: If True, reshards parameters post-forward pass to save memory.
        mp_policy: Controls mixed precision policy. If only dtype is provided, it will be used to cast all relevant parts of model. If None, no mixed precision is used
        cpu_offload: If True, enables CPU offloading of model parameters to reduce GPU memory usage.
        shard_on_largest_dim: If True, shards on the largest dimension of the parameter. By default FSDP shards on the first dimension, and if it is small will end up replicated on all ranks, which ends up increasing
            memory usage as world size increases.

    Note:
        It is recommended to specify specific modules to shard to avoid unnecessary sharding of all submodules, which has
        communication overhead.

    Note: modules_to_shard and shard_predicates are applied sequentially. If a module is specified in modules_to_shard, it will be sharded regardless of shard_predicates, and vice-versa

    Example:
        >>> model
            TransformerDecoder(
                (tok_embeddings): Embedding(128256, 4096)
                (layers): ModuleList(
                    (0-31): 32 x TransformerSelfAttentionLayer(
                    (attn): MultiHeadAttention(
                        (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
                        (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
                        (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
                        (output_proj): Linear(in_features=4096, out_features=4096, bias=False)
                        (pos_embeddings): RotaryPositionalEmbeddings()
                    )
                    ...
                )
                (output): Linear(in_features=4096, out_features=128256, bias=False)
            )
        >>> # You can either specify the module to shard as a name ("Linear") or the module type (torch.nn.Linear)
        >>> strategy = FSDP2Strategy(modules_to_shard=["TransformerSelfAttentionLayer", "Linear"])
    """

    modules_to_shard: Optional[
        Union[
            Literal["all"],
            Iterable[Union[str, Type[torch.nn.Module]]],
        ]
    ] = None
    shard_predicates: List[Callable[[str, torch.nn.Module], bool]] = field(
        default_factory=list
    )
    reshard_after_forward: Union[bool, int] = True
    mp_policy: Optional[Union[str, torch.dtype, MixedPrecisionPolicy]] = None
    cpu_offload: bool = False

    # experimental flag
    shard_on_largest_dim: bool = False


@dataclass
class TPStrategy(Strategy):
    """
    Dataclass representing Tensor Parallelism strategy. Specify the FSDP strategy for 2D parallelism setup.

    Args:
        tp_plan: The plan used to parallelize the module. See https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.parallelize_module for details.
        fsdp2_strategy (optional): fsdp2 strategy to configure 2D parallel strategy
    """

    tp_plan: Union[ParallelStyle, Dict[str, ParallelStyle]]
    fsdp2_strategy: Optional[FSDP2Strategy] = None


@dataclass
class TorchCompileParams:
    """
    Dataclass to store parameters for torch compile. See https://pytorch.org/docs/stable/generated/torch.compile.html for details.

    TNT specific args:
        recursive_module_types: list of module types to recursively compile. If not specified, applies compile to top-level module only.
            ex. ["TransformerCrossAttentionLayer", torch.nn.Linear] both work
    """

    fullgraph: bool = False
    dynamic: bool = False
    # pyre-ignore: Invalid type parameters. Uses PyTorch types.
    backend: Union[str, Callable] = "inductor"
    mode: Union[str, None] = None
    options: Optional[Dict[str, Union[str, int, bool]]] = None
    disable: bool = False

    # TNT specific params
    recursive_module_types: Collection[Union[str, Type[torch.nn.Module]]] = field(
        default_factory=list
    )


@dataclass
class ActivationCheckpointParams:
    """
    Dataclass to store parameters for activation checkpointing.

    Args:
        checkpoint_impl: type of checkpointing implementation to use
        check_fn: A lambda function which will be passed to each child submodule and return ``True`` or ``False`` depending on whether the submodule should be wrapped.
        auto_wrap_policy A policy to wrap model's submodules with AC. Note that if this is specified, it takes precedence over ``check_fn``.
    """

    checkpoint_impl: CheckpointImpl
    check_fn: Callable[[torch.nn.Module], bool] = lambda _: True
    auto_wrap_policy: Optional[Callable[[torch.nn.Module, bool, int], bool]] = None
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    context_fn: Optional[Callable[[], Tuple[ContextManager, ContextManager]]] = None


def prepare_ddp(
    module: torch.nn.Module,
    device: torch.device,
    strategy: Optional[DDPStrategy] = None,
) -> DDP:
    """
    Utility to move a module to device and wrap in `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_.

    Args:
        module: module to be wrapped in DDP. If module has params on meta device, they will be materialized on the device prior to DDP wrapping
        device: device to which module will be moved
        strategy: an instance of :class:`~torchtnt.utils.prepare_module.DDPStrategy` which defines the settings of DDP APIs

    Examples::
        strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
        module = nn.Linear(1, 1)
        device = torch.device("cuda")
        ddp_module = prepare_ddp(module, device, strategy)
    """
    strategy = strategy if strategy is not None else DDPStrategy()
    # wrap module in DDP
    device_ids = None
    if device.type == "cuda":
        device_ids = [device.index]
    params_dict = asdict(strategy)
    # remove ddp comm hook variables from params dict
    del params_dict["comm_state"]
    del params_dict["comm_hook"]

    materialize_meta_params(module, device)

    # now move rest of module to device
    module = module.to(device)

    # remove sync batch norm from params dict before converting module
    del params_dict["sync_batchnorm"]
    if strategy.sync_batchnorm:
        if device.type == "cuda":
            module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
        else:
            rank_zero_warn(
                f"SyncBatchNorm layers only work with GPU modules. Skipping the conversion because the device type is {device.type}."
            )

    module = DDP(module, device_ids=device_ids, **params_dict)
    if strategy.comm_hook:
        module.register_comm_hook(state=strategy.comm_state, hook=strategy.comm_hook)
    return module


def prepare_fsdp(
    module: torch.nn.Module,
    device: torch.device,
    strategy: Optional[FSDPStrategy] = None,
) -> FSDP:
    """
    Utility to move a module to device and wrap in `FullyShardedDataParallel <https://pytorch.org/docs/stable/fsdp.html>`_.

    Args:
        module: module to be wrapped in FSDP
        device: device to which module will be moved
        strategy: an instance of :class:`~torchtnt.utils.prepare_module.FSDPStrategy` which defines the settings of FSDP APIs

    Examples::
        strategy = FSDPStrategy(limit_all_gathers=True)
        module = nn.Linear(1, 1)
        device = torch.device("cuda")
        fsdp_module = prepare_fsdp(module, device, strategy)
    """
    strategy = strategy if strategy is not None else FSDPStrategy()

    # we use __dict__ and not asdict() here because asdict() is recursively applied on nested objects
    params_dict = strategy.__dict__.copy()

    # extract params to set state dict type
    state_dict_type = params_dict.pop("state_dict_type")
    state_dict_config = params_dict.pop("state_dict_config")
    optim_state_dict_config = params_dict.pop("optim_state_dict_config")

    # wrap module in FSDP
    module = FSDP(
        module,
        device_id=device,
        **params_dict,
    )

    if state_dict_type:
        FSDP.set_state_dict_type(
            module, state_dict_type, state_dict_config, optim_state_dict_config
        )
    return module


def prepare_fsdp2(
    module: torch.nn.Module,
    device: torch.device,
    strategy: Optional[FSDP2Strategy] = None,
    global_mesh: Optional[GlobalMeshCoordinator] = None,
) -> torch.nn.Module:
    """
    Utility to move a module to device and wrap in `FSDP2 <https://pytorch.org/docs/2.6/distributed.fsdp.fully_shard.html>`_

    Args:
        module: module to be wrapped in FSDP
        device: device to which module will be moved
        strategy: an instance of :class:`~torchtnt.utils.prepare_module.FSDP2Strategy` which defines the settings of FSDP APIs
        global_mesh: an instance of :class:`~torchtnt.utils.device_mesh.GlobalMeshCoordinator` which defines the global mesh topology.
            If not provided, a 1D default mesh will be created covering the entire world size.
    """
    strategy = strategy or FSDP2Strategy()

    # prepare kwargs for fully_shard api
    mesh: DeviceMesh
    if global_mesh is None:
        pg = dist.distributed_c10d._get_default_group()
        mesh = init_device_mesh(device.type, mesh_shape=(pg.size(),))
    else:
        mesh = global_mesh.dp_mesh

    fsdp_kwargs: Dict[str, Any] = {
        "mesh": mesh,  # TODO we only configure 1D mesh for now, look into supporting HSDP
        "reshard_after_forward": strategy.reshard_after_forward,
    }
    if strategy.cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()
    if (mp_policy := strategy.mp_policy) is not None:
        if isinstance(mp_policy, MixedPrecisionPolicy):
            mp_policy = _check_and_convert_mp_policy_dtypes(mp_policy)
            fsdp_kwargs["mp_policy"] = mp_policy
        elif isinstance(mp_policy, str):
            dtype = convert_precision_str_to_dtype(mp_policy)
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=dtype,
                reduce_dtype=dtype,
                output_dtype=dtype,
            )
        elif isinstance(mp_policy, torch.dtype):
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=mp_policy,
                reduce_dtype=mp_policy,
                output_dtype=mp_policy,
            )
    if strategy.shard_on_largest_dim:

        # From the docs: https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
        # "If sharding on a nonzero dim, we currently require even sharding, i.e. the tensor dim size on that dim must be divisible by the FSDP shard mesh size."

        # So we shard on a candidate nonzero dim only when it's divisible by the fsdp world size

        def _shard_placement_fn(param: torch.nn.Parameter) -> Optional[Shard]:
            largest_dim_size = max(param.shape)
            idx = param.shape.index(largest_dim_size)
            if idx != 0 and largest_dim_size % mesh.size() != 0:
                # not divisible, so we return None to shard on default dim 0
                return None
            return Shard(idx)

        fsdp_kwargs["shard_placement_fn"] = _shard_placement_fn

    # parse out the modules_to_shard argument
    modules_to_shard = strategy.modules_to_shard

    shard_all = modules_to_shard == "all"
    shard_module_names: Set[str] = set()
    shard_module_types: Tuple[Type[torch.nn.Module], ...] = ()
    if not shard_all and modules_to_shard is not None:
        assert (
            type(modules_to_shard) is not str
        ), f"modules_to_shard must be an iterable of modules or 'all', got {shard_all}"

        for item in none_throws(modules_to_shard):
            if isinstance(item, str):
                shard_module_names.add(item)
            else:
                shard_module_types = shard_module_types + (item,)

    # apply the fsdp2 sharding bottoms up
    num_layers_sharded = 0
    for n, m in reversed(list(module.named_modules())):
        if shard_all:
            # fully_shard does not support containers that do not implement forward
            if not isinstance(m, (torch.nn.ModuleList, torch.nn.ModuleDict)):
                fully_shard(m, **fsdp_kwargs)
                num_layers_sharded += 1
        elif (
            isinstance(m, shard_module_types) or type(m).__name__ in shard_module_names
        ):
            # if m exists in shard_module_types, then shard it
            fully_shard(m, **fsdp_kwargs)
            num_layers_sharded += 1
        elif len(strategy.shard_predicates) > 0:
            # if shard_predicates is not empty, then check if any of the conditions are true
            for predicate in strategy.shard_predicates:
                if predicate(n, m):
                    fully_shard(m, **fsdp_kwargs)
                    num_layers_sharded += 1
                    break

    if num_layers_sharded == 0:
        raise ValueError(
            "No layer modules were sharded with fsdp2. Please check if shard conditions are working as expected."
        )

    # shard the top level model, so that all params are moved off cpu to gpu
    if not _is_fsdp2_module(module):
        # disable reshard_after_forward for top level module
        # as result is DTensor which may be incompatible with
        # certain loss computation
        root_kwargs = deepcopy(fsdp_kwargs)
        root_kwargs["reshard_after_forward"] = False

        fully_shard(module, **root_kwargs)

    # materialized sharded meta weights to device
    materialize_meta_params(module, device)

    return module


def apply_ac(
    module: torch.nn.Module, activation_checkpoint_params: ActivationCheckpointParams
) -> None:
    """
    Applies activation checkpointing in-place.

    Args:
        module: module to apply activation checkpointing on
        activation_checkpoint_params: params to configure the activation checkpointing
    """
    checkpoint_impl = activation_checkpoint_params.checkpoint_impl
    check_fn = activation_checkpoint_params.check_fn
    auto_wrap_policy = activation_checkpoint_params.auto_wrap_policy
    context_fn = activation_checkpoint_params.context_fn
    additional_params = {}
    if context_fn:
        additional_params["context_fn"] = context_fn
    custom_checkpoint_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=checkpoint_impl,
        **additional_params,
    )
    apply_activation_checkpointing(
        module,
        checkpoint_wrapper_fn=custom_checkpoint_wrapper,
        check_fn=check_fn,
        auto_wrap_policy=auto_wrap_policy,
    )


def apply_torch_compile(
    module: torch.nn.Module,
    torch_compile_params: TorchCompileParams,
) -> None:
    """
    Applies torch.compile in-place on a given module.

    Args:
        module: module to apply torch.compile on
        torch_compile_params: params to configure the torch.compile
    """
    recursive_module_types = torch_compile_params.recursive_module_types
    params_dict = asdict(torch_compile_params)
    # remove recursive_module_types from params dict as we pass this directly to torch.compile
    params_dict.pop("recursive_module_types")
    try:
        # use in-place compile to avoid altering the state_dict keys

        if len(recursive_module_types) == 0:
            # compile only top-level module
            module.compile(**params_dict)
        else:
            # compile submodules recursively based on recursive_module_types

            # 1) separate str and torch.nn.Module types from recursive_module_types
            module_names: Set[str] = set()
            module_types: Tuple[Type[torch.nn.Module], ...] = ()
            for v in recursive_module_types:
                if isinstance(v, str):
                    module_names.add(v)
                else:
                    module_types = module_types + (v,)

            # 2) apply torch.compile recursively
            for m in reversed(list(module.modules())):
                if isinstance(m, module_types) or type(m).__name__ in module_names:
                    m.compile(**params_dict)
    except AttributeError:
        rank_zero_warn(
            "Please install PyTorch nightlies to use in-place compile to avoid altering the state_dict keys when checkpointing. Skipping torch compile."
        )


class FSDPOptimizerWrapper:
    """
    Wrapper for FSDP optimizer to call specific FSDP optimizer state checkpointing APIs.
    """

    def __init__(
        self, module: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> None:
        self.module = module
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        optim_state_dict = FSDP.optim_state_dict(self.module, self.optimizer)
        return optim_state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        optim_state_dict = FSDP.optim_state_dict_to_load(
            self.module, self.optimizer, state_dict
        )
        self.optimizer.load_state_dict(optim_state_dict)


class FSDP2OptimizerWrapper:
    """
    Wrapper for FSDP2 optimizer which uses distributed state dict APIs.
    """

    def __init__(
        self, module: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> None:
        self.module = module
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        return get_optimizer_state_dict(self.module, self.optimizer)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_optimizer_state_dict(self.module, self.optimizer, state_dict)


def _is_fsdp_module(module: torch.nn.Module) -> bool:
    """
    Checks if a module is wrapped in FSDP or FSDP2
    """
    return _is_fsdp1_module(module) or _is_fsdp2_module(module)


def _is_fsdp1_module(module: torch.nn.Module) -> bool:
    """
    Checks if a module is sharded by original FSDP
    """
    return isinstance(module, FSDP)


def _is_fsdp2_module(module: torch.nn.Module) -> bool:
    """
    Checks if a module is sharded by FSDP2
    """
    maybe_composable_state = _get_module_state(module)
    if maybe_composable_state is not None:
        return isinstance(maybe_composable_state, FSDPState)

    return False


def prepare_module(
    module: torch.nn.Module,
    device: torch.device,
    *,
    strategy: Optional[Union[Strategy, str]] = None,
    torch_compile_params: Optional[TorchCompileParams] = None,
    activation_checkpoint_params: Optional[ActivationCheckpointParams] = None,
    enable_compiled_autograd: bool = False,
    global_mesh: Optional[GlobalMeshCoordinator] = None,
) -> torch.nn.Module:
    """
    Utility to move a module to device, set up parallelism (None, DDP, FSDP, HSDP, TP), activation checkpointing and compile.
    This function acts as a dispatcher to choose between 1D and 2D parallelism setup, depending on the strategy used.

    Args:
        module: module to be used.
        device: device to which module will be moved.
        strategy: the data parallelization strategy to be used. if a string, must be one of ``ddp``, ``fsdp``, or ``noop``.
        torch_compile_params: params for Torch compile https://pytorch.org/docs/stable/generated/torch.compile.html.
        activation_checkpoint_params: params for enabling activation checkpointing.
        enable_compiled_autograd: if True, `compiled_autograd` will be used to compile the backward, this is an experimental flag.
        global_mesh: an instance of :class:`~torchtnt.utils.device_mesh.GlobalMeshCoordinator` which defines the global mesh topology.
    """
    if isinstance(strategy, TPStrategy):
        if global_mesh is None:
            raise ValueError(
                "TPStrategy expects global_mesh (GlobalMeshCoordinator) to be defined. Got None."
            )
        return _prepare_module_2d(
            module,
            device,
            strategy=strategy,
            global_mesh=global_mesh,
            torch_compile_params=torch_compile_params,
            activation_checkpoint_params=activation_checkpoint_params,
        )

    return _prepare_module_1d(
        module,
        device,
        strategy=strategy,
        torch_compile_params=torch_compile_params,
        activation_checkpoint_params=activation_checkpoint_params,
        enable_compiled_autograd=enable_compiled_autograd,
        global_mesh=global_mesh,
    )


def _prepare_module_1d(
    module: torch.nn.Module,
    device: torch.device,
    *,
    strategy: Optional[Union[Strategy, str]] = None,
    torch_compile_params: Optional[TorchCompileParams] = None,
    activation_checkpoint_params: Optional[ActivationCheckpointParams] = None,
    enable_compiled_autograd: bool = False,
    global_mesh: Optional[GlobalMeshCoordinator] = None,
) -> torch.nn.Module:
    """
    Utility to move a module to device, set up 1D parallelism (None, DDP, FSDP), activation checkpointing and compile.

    Args:
        module: module to be used.
        device: device to which module will be moved.
        strategy: the data parallelization strategy to be used. if a string, must be one of ``ddp``, ``fsdp``, or ``noop``.
        torch_compile_params: params for Torch compile https://pytorch.org/docs/stable/generated/torch.compile.html.
        activation_checkpoint_params: params for enabling activation checkpointing.
        enable_compiled_autograd: if True, `compiled_autograd` will be used to compile the backward, this is an experimental flag.
        global_mesh: an instance of :class:`~torchtnt.utils.device_mesh.GlobalMeshCoordinator` which defines the global mesh topology.
            Only pass here if wanting to configure HSDP setup with FSDP2
    """

    if strategy:
        if not isinstance(strategy, str) and not isinstance(strategy, Strategy):
            raise ValueError(
                f"Unknown strategy received: {strategy}. Expect either str (one of 'ddp', 'fsdp', or 'noop') or Strategy dataclass"
            )

        if isinstance(strategy, str):
            strategy = convert_str_to_strategy(strategy)
        if isinstance(strategy, DDPStrategy):
            if (
                torch_compile_params
                and strategy.static_graph is True
                and not is_torch_version_geq("2.1.0")
            ):
                raise RuntimeError(
                    "Torch version >= 2.1.0 required for Torch compile + DDP with static graph"
                )

            if enable_compiled_autograd:
                if not torch_compile_params:
                    raise RuntimeError(
                        "Compiled autograd should only be used when the module is compiled."
                    )
                try:
                    from torch._dynamo.trace_rules import LEGACY_MOD_INLINELIST

                    LEGACY_MOD_INLINELIST.add("torch.nn.parallel.distributed")
                except ImportError:
                    pass
                # This has to be set before DDP wrapping
                torch._dynamo.config.optimize_ddp = "python_reducer"
            module = prepare_ddp(module, device, strategy)
        elif isinstance(strategy, FSDPStrategy):
            if torch_compile_params and strategy.use_orig_params is False:
                # as stated here https://pytorch.org/get-started/pytorch-2.0/
                raise RuntimeError(
                    "Torch compile requires FSDPStrategy's use_orig_params to be True, since AOTAutograd needs to be aware of the original parameters"
                )
            module = prepare_fsdp(module, device, strategy)
        elif isinstance(strategy, FSDP2Strategy):
            module = prepare_fsdp2(module, device, strategy, global_mesh=global_mesh)
    else:
        # materialize any meta device params
        materialize_meta_params(module=module, device=device)
        # then move entire module to device
        module = module.to(device)

    if activation_checkpoint_params:
        apply_ac(module, activation_checkpoint_params)

    if torch_compile_params:
        apply_torch_compile(module, torch_compile_params)

    return module


def _prepare_module_2d(
    module: torch.nn.Module,
    device: torch.device,
    *,
    strategy: TPStrategy,
    global_mesh: GlobalMeshCoordinator,
    torch_compile_params: Optional[TorchCompileParams] = None,
    activation_checkpoint_params: Optional[ActivationCheckpointParams] = None,
) -> torch.nn.Module:
    """
    Utility to move a module to device, set up 2D parallelism (FSDP / TP / HSDP), activation checkpointing and compile.

    Order of composability is TP -> AC -> compile -> fsdp2.

    Args:
        module: module to be used.
        device: device to which module will be moved.
        strategy: the TP parallelization strategy to be used.
        global_mesh: an instance of :class:`~torchtnt.utils.device_mesh.GlobalMeshCoordinator` which defines the global mesh topology.
        torch_compile_params: params for Torch compile https://pytorch.org/docs/stable/generated/torch.compile.html.
        activation_checkpoint_params: params for enabling activation checkpointing.
    """

    # 1) apply TP
    parallelize_module(module, global_mesh.tp_mesh, parallelize_plan=strategy.tp_plan)

    # 2) apply AC if specified
    if activation_checkpoint_params:
        apply_ac(module, activation_checkpoint_params)

    # 3) apply torch.compile is specified
    if torch_compile_params:
        apply_torch_compile(module, torch_compile_params)

    # 4) apply data parallel / HSDP sharding (via FSDP2 apis) if specified in TPStrategy
    if (fsdp2_strategy := strategy.fsdp2_strategy) is not None:
        prepare_fsdp2(module, device, fsdp2_strategy, global_mesh)
    else:
        # prepare_fsdp2 will handle materializing meta weights
        # so if fsdp2strategy isn't used, we do it manually here
        materialize_meta_params(module, device)

    return module


def convert_str_to_strategy(
    strategy: str,
) -> Union[DDPStrategy, FSDPStrategy, FSDP2Strategy, NOOPStrategy]:
    """
    Converts strategy as a string to a default instance of the Strategy dataclass.

    Args:
        strategy: string specifying the distributed strategy to use

    Raises:
        ValueError if an invalid strategy string is passed.

    """
    string_to_strategy_mapping = {
        "ddp": DDPStrategy(),
        "fsdp": FSDPStrategy(),
        "fsdp2": FSDP2Strategy(),
        "noop": NOOPStrategy(),
    }

    if strategy not in string_to_strategy_mapping:
        raise ValueError(
            f"Strategy {strategy} not supported. Please use one of {list(string_to_strategy_mapping.keys())}"
        )
    return string_to_strategy_mapping[strategy]


def on_meta_device(module: torch.nn.Module) -> bool:
    try:
        return next(module.parameters(recurse=False)).device.type == "meta"
    except StopIteration:
        return False


def materialize_meta_params(module: torch.nn.Module, device: torch.device) -> None:
    """
    Materialize meta device parameters to the given device.

    Args:
        module: module to be used.
        device: device to which module will be moved.
    """
    for name, submodule in module.named_modules():
        if on_meta_device(submodule):
            rank_zero_info(f"{name} is on meta device, intializing on device {device}")
            submodule.to_empty(device=device, recurse=False)


def _check_and_convert_mp_policy_dtypes(
    mp_policy: MixedPrecisionPolicy,
) -> MixedPrecisionPolicy:
    """
    Converts precision strings to torch.dtype and validates that all dtypes are of type torch.dtype.
    Returns new MixedPrecisionPolicy as its attributes are frozen (cannot assign new values to fields)
    """

    dtypes = (mp_policy.param_dtype, mp_policy.reduce_dtype, mp_policy.output_dtype)
    dtypes = filter(None, dtypes)
    for dtype in dtypes:
        if not isinstance(dtype, (str, torch.dtype)):
            raise ValueError(
                f"MixedPrecisionPolicy requires all dtypes to be torch.dtype or string. Got dtype={dtype} with type {type(dtype)}"
            )

    param_dtype = mp_policy.param_dtype
    reduce_dtype = mp_policy.reduce_dtype
    output_dtype = mp_policy.output_dtype
    if isinstance(mp_policy.param_dtype, str):
        param_dtype = convert_precision_str_to_dtype(mp_policy.param_dtype)
    if isinstance(mp_policy.reduce_dtype, str):
        reduce_dtype = convert_precision_str_to_dtype(mp_policy.reduce_dtype)
    if isinstance(mp_policy.output_dtype, str):
        output_dtype = convert_precision_str_to_dtype(mp_policy.output_dtype)

    new_mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        cast_forward_inputs=mp_policy.cast_forward_inputs,
    )

    return new_mp_policy


def get_module_state_dict(
    module: torch.nn.Module, rank0_only: bool = False
) -> Dict[str, Any]:
    """
    Given a module, return a state dict that can be loaded into a CPU instance of the module. This requires different implementation depending on strategy:
    - If FSDP, we need to gather all the sharded parameters and offload state dict to CPU in order to avoid OOM.
    - If DDP, we need to unwrap the module to avoid extra state_dict prefix
    - Otherwise, we can just return the state dict as is

    Args:
        module: module to be used.
        rank0_only: This flag only works for FSDP. If True, only rank 0 will return the state dict. Other ranks will return an empty dict.
            For DDP or no strategy case, we don't move the state dice to CPU -- it can be loaded directly into the module.

    Note: Even if the state_dict parameters are on GPU, it can still be loaded into a CPU module.
    """
    logger.info("Generating module state dict")

    # TODO: Add support for FSDP2
    if isinstance(module, FSDP):
        state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)
        with FSDP.state_dict_type(module, _StateDictType.FULL_STATE_DICT, state_cfg):
            return module.state_dict()

    if rank0_only:
        logger.warning(
            "Provided rank0_only=True, but this is no-op for DDP or no strategy. Returning state dict in module's device."
        )

    if isinstance(module, DDP):
        module = module.module

    state_dict = module.state_dict()

    return state_dict
