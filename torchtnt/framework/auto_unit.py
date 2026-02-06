# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import contextlib
import logging
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    cast,
    ContextManager,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
from pyre_extensions import none_throws
from torch.distributed.fsdp import FSDPModule, FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel.loss import loss_parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import SWALR
from torchtnt.framework.state import ActivePhase, EntryPoint, State
from torchtnt.framework.unit import EvalUnit, PredictUnit, TPredictData, TrainUnit
from torchtnt.framework.utils import get_timing_context
from torchtnt.utils.device import copy_data_to_device
from torchtnt.utils.device_mesh import GlobalMeshCoordinator
from torchtnt.utils.env import init_from_env
from torchtnt.utils.lr_scheduler import TLRScheduler
from torchtnt.utils.precision import (
    convert_precision_str_to_dtype,
    get_grad_scaler_from_precision,
    GradScaler,
)
from torchtnt.utils.prepare_module import (
    _is_fsdp1_module,
    _is_fsdp2_module,
    ActivationCheckpointParams,
    FSDPStrategy,
    prepare_fsdp,
    prepare_module,
    Strategy,
    TorchCompileParams,
    TPStrategy,
)
from torchtnt.utils.swa import AveragedModel
from typing_extensions import Literal

_logger: logging.Logger = logging.getLogger(__name__)


TData = TypeVar("TData")


@dataclass
class SWALRParams:
    """
    Dataclass to store parameters for SWALR learning rate scheduler.
    See https://github.com/pytorch/pytorch/blob/main/torch/optim/swa_utils.py#L279
    for more details.

    Args:
        anneal_steps_or_epochs: number of steps or epochs to anneal the SWA Scheduler
        anneal_strategy: method for annealing, supports "linear" and "cos"
        swa_lrs: the learning rate value for all param groups together or separately for each group

        Note: Whether steps or epochs is used based on what `step_lr_interval` is set on the AutoUnit.
    """

    anneal_steps_or_epochs: int
    anneal_strategy: str = "linear"
    swa_lrs: Union[List[float], float] = 0.05


@dataclass
class SWAParams:
    """
    Dataclass to store parameters for stochastic weight averaging.

    Args:
        warmup_steps_or_epochs: number of steps or epochs before starting SWA
        step_or_epoch_update_freq: number of steps or epochs between each SWA update
        use_buffers: if ``True``, it will compute running averages for
            both the parameters and the buffers of the model. (default: ``True``)
            This will update activation statistics for Batch Normalization. This is an
            alternative to calling `torch.optim.swa_utils.update_bn` post-training.
        averaging_method: whether to use SWA or EMA to average model weights
        ema_decay:  the exponential decay applied to the averaged parameters. This param
            is only needed for EMA, and is ignored otherwise (for SWA).
        use_lit: if True, will use Lit EMA style by adjusting weight decay based on the
            number of updates. The EMA decay will start small and will approach the
            specified ema_decay as more updates occur. The ``averaging_method`` must be
            set to ema.
        swalr_params: params for SWA learning rate scheduler

        Note: Whether steps or epochs is used based on what `step_lr_interval` is set on the AutoUnit.

        Note: Only one of avg_fn, multi_avg_fn should be specified

    """

    warmup_steps_or_epochs: int
    step_or_epoch_update_freq: int
    use_buffers: bool = True
    averaging_method: Literal["ema", "swa"] = "ema"
    ema_decay: float = 0.999
    use_lit: bool = False
    swalr_params: Optional[SWALRParams] = None


@dataclass
class TrainStepResults:
    """
    Dataclass to store training step results.

    Args:
        loss: the loss computed in the ``compute_loss`` function
        total_grad_norm: total norm of the parameter gradients, if gradient norm clipping is enabled
        outputs: the outputs of the model forward pass
    """

    loss: torch.Tensor
    total_grad_norm: Optional[torch.Tensor]
    outputs: Any


class _ConfigureOptimizersCaller(ABCMeta):
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __call__(self, *args, **kwargs):
        x = super().__call__(*args, **kwargs)
        x.optimizer = None
        x.lr_scheduler = None
        x.swa_scheduler = None

        if x.training:
            x.optimizer, x.lr_scheduler = x.configure_optimizers_and_lr_scheduler(
                x.module
            )

            if x.swa_params and x.swa_params.swalr_params:
                swalr_params = x.swa_params.swalr_params
                x.swa_scheduler = SWALR(
                    optimizer=x.optimizer,
                    swa_lr=swalr_params.swa_lrs,
                    anneal_epochs=swalr_params.anneal_steps_or_epochs,
                    anneal_strategy=swalr_params.anneal_strategy,
                )

        return x


class _AutoUnitMixin(Generic[TData]):
    """
    A mixin to share initialization of shared attributes and introduce prefetching.
    """

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        device: Optional[torch.device] = None,
        precision: Optional[Union[str, torch.dtype]] = None,
        detect_anomaly: Optional[bool] = None,
        torch_compile_params: Optional[TorchCompileParams] = None,
        enable_prefetch: bool = True,
    ) -> None:
        super().__init__()

        self.device: torch.device = device or init_from_env()
        self.precision: Optional[torch.dtype] = (
            convert_precision_str_to_dtype(precision)
            if isinstance(precision, str)
            else precision
        )

        self.detect_anomaly = detect_anomaly
        if torch_compile_params is not None and not torch_compile_params.disable:
            # torch compile is not compatible with detect anomaly
            # so we disable detect anomaly if torch compile is enabled
            self.detect_anomaly = None
            _logger.warning("torch.compile is enabled, so detect_anomaly is disabled")

        # create autocast context based on precision and device type
        self.maybe_autocast_precision = torch.autocast(
            device_type=self.device.type,
            dtype=self.precision,
            enabled=self.precision is not None,
        )

        # main stream responsible for computation on the device
        self._default_stream: Optional[torch.cuda.streams.Stream] = (
            torch.cuda.current_stream()
            if (self.device.type == "cuda" and enable_prefetch)
            else None
        )
        # cuda stream to use for moving data to device
        self._prefetch_stream: Optional[torch.cuda.streams.Stream] = (
            torch.cuda.Stream()
            if (self.device.type == "cuda" and enable_prefetch)
            else None
        )
        # dict mapping phase to whether the next batch which has been prefetched for that phase and is ready to be used
        self._phase_to_next_batch: dict[ActivePhase, Optional[TData]] = {
            ActivePhase.TRAIN: None,
            ActivePhase.EVALUATE: None,
            ActivePhase.PREDICT: None,
        }

        # dict mapping phase to whether the next batch for that phase has been prefetched and is ready to be used
        self._phase_to_prefetched: dict[ActivePhase, bool] = {
            ActivePhase.TRAIN: False,
            ActivePhase.EVALUATE: False,
            ActivePhase.PREDICT: False,
        }
        # whether the current batch is the last train batch
        self._is_last_batch: bool = False
        self._enable_prefetch = enable_prefetch

    def move_data_to_device(
        self,
        state: State,
        data: TData,
        non_blocking: bool,
    ) -> TData:
        """
        The user can override this method with custom code to copy data to device. This will be called at the start of every ``train_step``/``eval_step``/``predict_step``.
        By default this uses the utility function :py:func:`~torchtnt.utils.copy_data_to_device`.

        If on GPU, this method will be called on a separate CUDA stream.

        Args:
            state: a State object which is passed from the ``train_step``/``eval_step``/``predict_step``
            data: a batch of data which is passed from the ``train_step``/``eval_step``/``predict_step``
            non_blocking: parameter to pass to ``torch.tensor.to``

        Returns:
            A batch of data which is on the device

        Note:
            If overriding, ensure that tensors are recorded on the compute stream to avoid the cuda cache allocator from
            overwriting the underlying data before the compute stream has a chance to use it. If using `copy_data_to_device`,
            you can pass `stream_to_record=self._default_stream` as an argument.
        """
        return copy_data_to_device(
            data,
            self.device,
            non_blocking=non_blocking,
            stream_to_record=self._default_stream,
        )

    def _prefetch_next_batch(self, state: State, data_iter: Iterator[TData]) -> None:
        """Prefetch the next batch on a separate CUDA stream."""
        active_phase = state.active_phase
        phase = state.active_phase.name.lower()
        try:
            with get_timing_context(
                state, f"{self.__class__.__name__}.{phase}.next(data_iter)"
            ):
                next_batch = next(data_iter)
        except StopIteration:
            self._phase_to_next_batch[active_phase] = None
            self._is_last_batch = True
            return

        non_blocking = bool(
            self.device.type == "cuda" and self._phase_to_prefetched[active_phase]
        )

        # if on cpu, self._prefetch_stream is None so the torch.cuda.stream call is a no-op
        with torch.cuda.stream(self._prefetch_stream), get_timing_context(
            state, f"{self.__class__.__name__}.{phase}.move_data_to_device"
        ):
            self._phase_to_next_batch[active_phase] = self.move_data_to_device(
                state,
                next_batch,
                non_blocking=non_blocking,
            )

    def _get_next_batch(self, state: State, data: Iterator[TData]) -> TData:
        if not self._enable_prefetch:
            batch = next(data)
            return self.move_data_to_device(state, batch, non_blocking=True)

        active_phase = state.active_phase
        if not self._phase_to_prefetched[active_phase]:
            self._prefetch_next_batch(state, data)
            self._phase_to_prefetched[active_phase] = True

        if self._prefetch_stream:
            with get_timing_context(state, f"{self.__class__.__name__}.wait_stream"):
                # wait on the CUDA stream to complete the host to device copy
                torch.cuda.current_stream().wait_stream(self._prefetch_stream)

        # get the next batch which was stored by _prefetch_next_batch
        batch = self._phase_to_next_batch[active_phase]
        if batch is None:
            self._phase_to_prefetched[active_phase] = False
            self._is_last_batch = False
            raise StopIteration

        # prefetch the next batch
        self._prefetch_next_batch(state, data)

        return batch


class AutoPredictUnit(_AutoUnitMixin[TPredictData], PredictUnit[TPredictData]):
    def __init__(
        self,
        *,
        module: torch.nn.Module,
        device: Optional[torch.device] = None,
        strategy: Optional[Union[Strategy, str]] = None,
        precision: Optional[Union[str, torch.dtype]] = None,
        torch_compile_params: Optional[TorchCompileParams] = None,
        detect_anomaly: Optional[bool] = None,
        enable_prefetch: bool = False,
        global_mesh: Optional[GlobalMeshCoordinator] = None,
    ) -> None:
        """
        AutoPredictUnit is a convenience for users who are running inference and would like to have certain features handled for them, such as:
        - Moving data to the correct device.
        - Running inference under a mixed precision context.
        - Handling data parallel replication, especially if the module cannot fit on a single device using FullyShardedDataParallel.
        - Profiling the data transfer to device and forward pass.
        - Interleaving moving the next batch to the device with running the module's forward pass on the current batch.

        Additionally, the AutoPredictUnit offers an optional hook ``on_predict_step_end`` to further post-process module outputs if desired.

        Then use with the :py:func:`~torchtnt.framework.predict` entry point.

        For more advanced customization, directly use the :class:`~torchtnt.framework.unit.PredictUnit` interface.

        Args:
            module: module to be used during prediction.
            device: the device to be used.
            precision: the precision to use in training, as either a string or a torch.dtype.
            strategy: the data parallelization strategy to be used. if a string, must be one of ``ddp`` or ``fsdp``.
            torch_compile_params: params for Torch compile https://pytorch.org/docs/stable/generated/torch.compile.html
            detect_anomaly: whether to enable anomaly detection for the autograd engine https://pytorch.org/docs/stable/autograd.html#anomaly-detection
            global_mesh: an instance of :class:`~torchtnt.utils.device_mesh.GlobalMeshCoordinator` which defines the global mesh topology. Needed to configure TP or 2D parallelism strategies.

        Note:
            Torch compile support is only available in PyTorch 2.0 or higher.
        """
        super().__init__(
            module=module,
            device=device,
            precision=precision,
            torch_compile_params=torch_compile_params,
            detect_anomaly=detect_anomaly,
            enable_prefetch=enable_prefetch,
        )
        self.module: torch.nn.Module = prepare_module(
            module,
            self.device,
            strategy=strategy,
            torch_compile_params=torch_compile_params,
            global_mesh=global_mesh,
        )

    def predict_step(self, state: State, data: TPredictData) -> Any:
        # if detect_anomaly is true, run forward pass under detect_anomaly context
        detect_anomaly = self.detect_anomaly
        maybe_detect_anomaly = (
            torch.autograd.set_detect_anomaly(detect_anomaly)
            if detect_anomaly is not None
            else contextlib.nullcontext()
        )

        with self.maybe_autocast_precision, maybe_detect_anomaly:
            with get_timing_context(state, f"{self.__class__.__name__}.forward"):
                outputs = self.module(data)

        step = self.predict_progress.num_steps_completed
        self.on_predict_step_end(state, data, step, outputs)
        return outputs

    def on_predict_step_end(
        self,
        state: State,
        data: TPredictData,
        step: int,
        outputs: Any,
    ) -> None:
        """
        This will be called at the end of every ``predict_step`` before returning. The user can implement this method with code to update and log their metrics,
        or do anything else.

        Args:
            state: a State object which is passed from the ``predict_step``
            data: a batch of data which is passed from the ``predict_step``
            step: how many ``predict_step`` s have been completed
            outputs: the outputs of the model forward pass
        """
        pass

    def get_next_predict_batch(
        self, state: State, data_iter: Iterator[TPredictData]
    ) -> Union[Iterator[TPredictData], TPredictData]:
        # Override the default behavior from PredictUnit in order to enable prefetching if possible.
        if self._predict_step_requires_iterator:
            return data_iter
        return self._get_next_batch(state, data_iter)


class AutoUnit(
    _AutoUnitMixin[TData],
    TrainUnit[TData],
    EvalUnit[TData],
    PredictUnit[TData],
    metaclass=_ConfigureOptimizersCaller,
):
    """
    The AutoUnit is a convenience for users who are training with stochastic gradient descent and would like to have model optimization
    and data parallel replication handled for them.
    The AutoUnit subclasses :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, and
    :class:`~torchtnt.framework.unit.PredictUnit` and implements the ``train_step``, ``eval_step``, and ``predict_step`` methods for the user.

    For the ``train_step`` it runs:

    - forward pass and loss computation
    - backward pass
    - optimizer step

    For the ``eval_step`` it only runs forward and loss computation.

    For the ``predict_step`` it only runs forward computation.

    To benefit from the AutoUnit, the user must subclass it and implement the ``compute_loss`` and ``configure_optimizers_and_lr_scheduler`` methods.
    Additionally, the AutoUnit offers these optional hooks:

    - ``on_train_step_end``
    - ``on_eval_step_end``
    - ``on_predict_step_end``

    The user can also override the LR step method, ``step_lr_scheduler``, in case they want to have custom logic.

    Then use with the :py:func:`~torchtnt.framework.train`, :py:func:`~torchtnt.framework.evaluate`, :py:func:`~torchtnt.framework.fit`, or
    :py:func:`~torchtnt.framework.predict` entry point as normal.

    For more advanced customization, directly use the :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`,
    and :class:`~torchtnt.framework.unit.PredictUnit` interfaces.

    Args:
        module: module to be used during training/evaluation.
        device: the device to be used.
        strategy: the data parallelization strategy to be used. if a string, must be one of ``ddp`` or ``fsdp``.
        step_lr_interval: whether to step lr_scheduler every step or every epoch. Defaults to every epoch.
        precision: the precision to use in training/evaluation (using automatic mixed precision), as either a string or a torch.dtype. Acceptable strings are ``'fp32'``, ``'fp16'``, and ``'bf16'``.
        gradient_accumulation_steps: how many batches to accumulate gradients over.
        detect_anomaly: whether to enable anomaly detection for the autograd engine https://pytorch.org/docs/stable/autograd.html#anomaly-detection
        clip_grad_norm: max norm of the gradients for clipping https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
        clip_grad_value: max value of the gradients for clipping https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html
        swa_params: params for stochastic weight averaging https://pytorch.org/docs/stable/optim.html#stochastic-weight-averaging (Please see note if using with FSDP)
        torch_compile_params: params for Torch compile https://pytorch.org/docs/stable/generated/torch.compile.html
        activation_checkpoint_params: params for enabling activation checkpointing
        training: if True, the optimizer and optionally LR scheduler will be created after the class is initialized.
        enable_compiled_autograd: if True, `compiled_autograd` will be used to compile the backward, this is an experimental flag.
        loss_backward_retain_graph:  If ``None`` or ``False``, the graph used to compute
            the grads will be freed during loss backward pass. Note that in nearly all cases setting
            this option to True is not needed and often can be worked around
            in a much more efficient way.
        enable_prefetch: if True, the data will be prefetched to the device before the next batch is loaded
        zero_grad_at_train_step_start: if True, the optimizer's gradients will be zeroed at the start of each train step, rather than at the end. Useful if you want to inspect/log the gradients via custom callback.
        global_mesh: an instance of :class:`~torchtnt.utils.device_mesh.GlobalMeshCoordinator` which defines the global mesh topology. Needed to configure TP or 2D parallelism strategies.
        enable_loss_parallel: if True, the loss will be computed in parallel across all ranks. This is only supported for TP strategy + cross entropy loss.

    Note:
        Certain strategies, like :class:`~torchtnt.utils.prepare_module.FSDPStrategy` also support mixed precision as an argument, so can be configured through that class as well.

    Note:
        If :class:`~torchtnt.utils.prepare_module.FSDPStrategy` and SWAParams are passed in, the swa model will be sharded with the same FSDP parameters.

    Note:
        Torch compile support is only available in PyTorch 2.0 or higher.

    Note:
        If using SWA with FSDP, the SWA model will be sharded with the same FSDP configuration as the original model. If you need the swa model's output in evaluation / prediction step,
        please call `self.swa_model(inputs, ...)` to ensure all hooks (especially for FSDP) are fired correctly.

    """

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        device: Optional[torch.device] = None,
        strategy: Optional[Union[Strategy, str]] = None,
        step_lr_interval: Literal["step", "epoch"] = "epoch",
        precision: Optional[Union[str, torch.dtype]] = None,
        gradient_accumulation_steps: int = 1,
        detect_anomaly: Optional[bool] = None,
        clip_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
        swa_params: Optional[SWAParams] = None,
        torch_compile_params: Optional[TorchCompileParams] = None,
        activation_checkpoint_params: Optional[ActivationCheckpointParams] = None,
        training: bool = True,
        enable_compiled_autograd: bool = False,
        loss_backward_retain_graph: Optional[bool] = None,
        enable_prefetch: bool = True,
        zero_grad_at_train_step_start: bool = False,
        global_mesh: Optional[GlobalMeshCoordinator] = None,
        enable_loss_parallel: bool = False,
    ) -> None:
        super().__init__(
            module=module,
            device=device,
            precision=precision,
            detect_anomaly=detect_anomaly,
            torch_compile_params=torch_compile_params,
            enable_prefetch=enable_prefetch,
        )

        if not gradient_accumulation_steps > 0:
            raise ValueError(
                f"gradient_accumulation_steps must be > 0. Got {gradient_accumulation_steps}"
            )

        self.swa_params: Optional[SWAParams] = swa_params
        self.swa_model: Optional[AveragedModel] = None
        if swa_params and training:
            module_for_swa = module
            skip_deepcopy = False
            if isinstance(strategy, FSDPStrategy):
                # must use exact same FSDPStrategy as original module
                # so each rank can computes EMA for its own local shard
                # since models are sharded identically if FSDP params are the same
                module_for_swa = prepare_fsdp(deepcopy(module), self.device, strategy)
                skip_deepcopy = True

            self.swa_model = AveragedModel(
                module_for_swa,
                device=self.device,
                use_buffers=swa_params.use_buffers,
                averaging_method=swa_params.averaging_method,
                ema_decay=swa_params.ema_decay,
                skip_deepcopy=skip_deepcopy,
                use_lit=swa_params.use_lit,
            )

        self.module: torch.nn.Module = prepare_module(
            module,
            self.device,
            strategy=strategy,
            torch_compile_params=torch_compile_params,
            activation_checkpoint_params=activation_checkpoint_params,
            enable_compiled_autograd=enable_compiled_autograd,
            global_mesh=global_mesh,
        )

        self.grad_scaler: Optional[GradScaler] = None
        if self.precision:
            self.grad_scaler = get_grad_scaler_from_precision(
                self.precision,
                is_fsdp1_module=_is_fsdp1_module(self.module),
            )

        self.step_lr_interval = step_lr_interval

        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        # create autocast context based on precision and device type

        self.enable_compiled_autograd = enable_compiled_autograd
        self.training = training
        self.loss_backward_retain_graph = loss_backward_retain_graph

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Optional[TLRScheduler] = None
        self.swa_scheduler: Optional[SWALR] = None

        self.zero_grad_at_train_step_start: bool = zero_grad_at_train_step_start
        # keep track of when to zero grad at train step start
        self._weight_updated_in_prev_step = False

        if enable_loss_parallel:
            if not isinstance(strategy, TPStrategy):
                raise ValueError(
                    "enable_loss_parallel is only supported with TPStrategy"
                )
        # pyre-fixme[24]: Attribute must be annotated.
        self.maybe_loss_parallel: Callable = (
            loss_parallel if enable_loss_parallel else contextlib.nullcontext
        )

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, torch.nn.Module):
            self._validate_module_attr(name, value)

        super().__setattr__(name, value)

    def _validate_module_attr(self, name: str, module: torch.nn.Module) -> None:
        """
        The AutoUnit is designed to manage the input model using the `self.module` attribute,
        which should not be reassigned. Additionally, if a subclass saves another attribute
        referencing the same model instance (wrapped or unwrapped), then the same instance will
        appear two times in the tracked_modules. This is problematic for checkpointing and handling
        of evaluation/training mode.
        """
        # First time the module attribute is set is in the AutoUnit's initialization
        if not hasattr(self, "module"):
            return

        # Value of self.module should not be changed after initialization
        if name == "module":
            _logger.warning(
                "The self.module attribute is managed by AutoUnit and is not meant to be reassigned."
            )
            return

        # Otherwise, double check that this is not a duplicate reference to the self.module instance
        managed_modules = [self.module]
        if isinstance(self.module, DDP) or isinstance(self.module, FSDP):
            managed_modules.append(self.module.module)

        if any(module is managed_module for managed_module in managed_modules):
            _logger.error(
                f"Attribute '{name}' of the custom TNT Unit stores a reference to the model managed"
                + "by AutoUnit. This is known to cause errors on checkpointing and model training "
                + "mode. Please remove this attribute and access the existing `self.module` instead."
            )

    @abstractmethod
    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, Optional[TLRScheduler]]:
        """
        The user should implement this method with their optimizer and learning rate scheduler construction code. This will be called upon initialization of
        the AutoUnit.

        Args:
            module: the module with which to construct optimizer and lr_scheduler

        Returns:
            A tuple containing optimizer and optionally the learning rate scheduler
        """
        ...

    @abstractmethod
    def compute_loss(self, state: State, data: TData) -> Tuple[torch.Tensor, Any]:
        """
        The user should implement this method with their loss computation. This will be called every ``train_step``/``eval_step``.

        Args:
            state: a State object which is passed from the ``train_step``/``eval_step``
            data: a batch of data which is passed from the ``train_step``/``eval_step``

        Returns:
            Tuple containing the loss and the output of the model

        Note:
            The module's forward pass must be run as part of this method.
        """
        ...

    def train_step(self, state: State, data: TData) -> Tuple[torch.Tensor, Any]:
        should_update_weights = (
            self.train_progress.num_steps_completed_in_epoch + 1
        ) % self.gradient_accumulation_steps == 0 or self._is_last_batch

        # zero the gradients if previous step updated weights
        if self._weight_updated_in_prev_step and self.zero_grad_at_train_step_start:
            self.zero_grad(state)
            self._weight_updated_in_prev_step = False

        # for pyre, assign to local variable
        module = self.module

        # if using gradient accumulation with either DDP or FSDP, when in a step where we will not update the weights,
        # run forward and backward in no_sync context
        # https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel.no_sync
        # https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.no_sync
        maybe_no_sync = (
            module.no_sync()
            if not should_update_weights
            and (isinstance(module, DDP) or isinstance(module, FSDP))
            else contextlib.nullcontext()
        )
        # fsdp2 has separate way of disabling gradient sync
        if _is_fsdp2_module(module):
            if not should_update_weights:
                cast(FSDPModule, module).set_requires_gradient_sync(False)
            elif should_update_weights and self.gradient_accumulation_steps > 1:
                # if gradient accumulation is used and it's time to update weights,
                # we need to re-enable gradient sync
                cast(FSDPModule, module).set_requires_gradient_sync(True)

        # if detect_anomaly is true, run forward and backward pass in detect_anomaly context
        detect_anomaly = self.detect_anomaly
        maybe_detect_anomaly = (
            torch.autograd.set_detect_anomaly(detect_anomaly)
            if detect_anomaly is not None
            else contextlib.nullcontext()
        )

        grad_scaler = self.grad_scaler
        with maybe_no_sync, maybe_detect_anomaly, self.maybe_loss_parallel():
            with self.maybe_autocast_precision:
                with get_timing_context(
                    state, f"{self.__class__.__name__}.compute_loss"
                ):
                    # Run the forward pass and compute the loss
                    loss, outputs = self.compute_loss(state, data)

            # normalize loss to account for gradient accumulation
            loss = self._normalize_loss_for_gradient_accumulation(loss)

            try:
                from torch._dynamo.utils import maybe_enable_compiled_autograd
            except ImportError:

                def maybe_enable_compiled_autograd(
                    val: bool,
                    # pyre-fixme[24]: Generic type `ContextManager` expects 1 type
                    #  parameter.
                ) -> ContextManager:
                    return contextlib.nullcontext()

            with maybe_enable_compiled_autograd(self.enable_compiled_autograd):
                if grad_scaler:
                    scaled_loss = grad_scaler.scale(loss)
                    with get_timing_context(
                        state, f"{self.__class__.__name__}.backward"
                    ):
                        scaled_loss.backward(
                            retain_graph=self.loss_backward_retain_graph
                        )
                else:
                    with get_timing_context(
                        state, f"{self.__class__.__name__}.backward"
                    ):
                        loss.backward(retain_graph=self.loss_backward_retain_graph)

        total_grad_norm = None
        if should_update_weights:
            total_grad_norm = self._update_weights(state)

        step = self.train_progress.num_steps_completed
        results = TrainStepResults(loss, total_grad_norm, outputs)
        self.on_train_step_end(state, data, step, results)
        return loss, outputs

    def _normalize_loss_for_gradient_accumulation(
        self, loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize the loss to account for gradient accumulation.
        Subclasses may override this to implement custom normalization strategies
        """
        return loss / self.gradient_accumulation_steps

    def on_train_step_end(
        self,
        state: State,
        data: TData,
        step: int,
        results: TrainStepResults,
    ) -> None:
        """
        This will be called at the end of every ``train_step`` before returning. The user can implement this method with code to update and log their metrics,
        or do anything else.

        Args:
            state: a State object which is passed from the ``train_step``
            data: a batch of data which is passed from the ``train_step``
            step: how many ``train_step`` s have been completed
            results: dataclass containing loss, total gradient norm, and outputs
        """
        pass

    def on_train_epoch_end(self, state: State) -> None:
        """
        Note: if overriding ``on_train_epoch_end``, remember to call ``super().on_train_epoch_end()``
        """
        if self.step_lr_interval == "epoch":
            # number of epochs is incremented before calling this, so we're offsetting by 1
            self._update_lr_and_swa(state, self.train_progress.num_epochs_completed - 1)

        self._is_last_batch = False

    def eval_step(self, state: State, data: TData) -> Tuple[torch.Tensor, Any]:
        with self.maybe_autocast_precision:
            # users must override this
            with get_timing_context(state, f"{self.__class__.__name__}.compute_loss"):
                loss, outputs = self.compute_loss(state, data)

        if state.entry_point == EntryPoint.FIT:
            step = self.train_progress.num_steps_completed
        else:
            step = self.eval_progress.num_steps_completed

        self.on_eval_step_end(state, data, step, loss, outputs)
        return loss, outputs

    def on_eval_step_end(
        self,
        state: State,
        data: TData,
        step: int,
        loss: torch.Tensor,
        outputs: Any,
    ) -> None:
        """
        This will be called at the end of every ``eval_step`` before returning. The user can implement this method with code to update and log their metrics,
        or do anything else.

        Args:
            state: a State object which is passed from the ``eval_step``
            data: a batch of data which is passed from the ``eval_step``
            step: how many steps have been completed (``train_step`` s when running fit and ``eval_step`` s when running evaluation)
            loss: the loss computed in the ``compute_loss`` function
            outputs: the outputs of the model forward pass
        """
        pass

    def predict_step(self, state: State, data: TData) -> Any:
        with self.maybe_autocast_precision:
            with get_timing_context(state, f"{self.__class__.__name__}.forward"):
                outputs = self.module(data)

        step = self.predict_progress.num_steps_completed
        # users can override this, by default this is a no-op
        with get_timing_context(
            state, f"{self.__class__.__name__}.on_predict_step_end"
        ):
            self.on_predict_step_end(state, data, step, outputs)
        return outputs

    def on_predict_step_end(
        self,
        state: State,
        data: TData,
        step: int,
        outputs: Any,
    ) -> None:
        """
        This will be called at the end of every ``predict_step`` before returning. The user can implement this method with code to update and log their metrics,
        or do anything else.

        Args:
            state: a State object which is passed from the ``predict_step``
            data: a batch of data which is passed from the ``predict_step``
            step: how many ``predict_step``s have been completed
            outputs: the outputs of the model forward pass
        """
        pass

    def step_lr_scheduler(self) -> None:
        """
        LR step method extracted to a method in case the user wants to override
        """
        none_throws(self.lr_scheduler).step()

    def zero_grad(self, state: State) -> None:
        """
        Zeroes the gradients of the module's parameters. You can override this if you want to log the gradients before zeroing them.

        Example of overriding:
            class CustomAutoUnit(MyAutoUnit):
                ...

                def zero_grad(self, state):
                    # log before zeroing gradients
                    super().zero_grad()
        """

        optimizer = none_throws(self.optimizer)
        with get_timing_context(
            state, f"{self.__class__.__name__}.optimizer_zero_grad"
        ):
            optimizer.zero_grad(set_to_none=True)

    def _update_weights(self, state: State) -> Optional[torch.Tensor]:
        """
        Updates weights of the module, handles clip gradient norm, etc.

        Returns total norm of the parameter gradients, if gradient norm clipping is enabled.
        """
        module = self.module
        optimizer = none_throws(self.optimizer)
        grad_scaler = self.grad_scaler
        # Run gradient clipping, optimizer step, and zero_grad
        clip_grad_norm = self.clip_grad_norm
        clip_grad_value = self.clip_grad_value
        if grad_scaler and (clip_grad_norm or clip_grad_value):
            # unscale the gradients of optimizer's assigned params in-place in preparation for gradient clipping
            with get_timing_context(state, f"{self.__class__.__name__}.grad_unscale"):
                grad_scaler.unscale_(optimizer)

        total_grad_norm = None
        # gradient norm clipping
        if clip_grad_norm:
            if isinstance(module, FSDP):
                with get_timing_context(
                    state, f"{self.__class__.__name__}.clip_grad_norm"
                ):
                    total_grad_norm = module.clip_grad_norm_(max_norm=clip_grad_norm)
            else:
                # strategy=None, DDP, and FSDP2 will work with this
                with get_timing_context(
                    state, f"{self.__class__.__name__}.clip_grad_norm"
                ):
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters=module.parameters(),
                        max_norm=clip_grad_norm,
                    )
                    # If sharded, collect the DTensor here
                    if isinstance(total_grad_norm, DTensor):
                        total_grad_norm = total_grad_norm.full_tensor()

        # gradient value clipping
        if clip_grad_value:
            with get_timing_context(
                state, f"{self.__class__.__name__}.clip_grad_value"
            ):
                torch.nn.utils.clip_grad_value_(
                    parameters=module.parameters(),
                    clip_value=clip_grad_value,
                )

        with get_timing_context(state, f"{self.__class__.__name__}.optimizer_step"):
            if grad_scaler:
                grad_scaler.step(optimizer)
                # update the scale for next iteration
                grad_scaler.update()
            else:
                optimizer.step()

        if self.zero_grad_at_train_step_start:
            # mark that weights were updated in this step
            # so in next step we know to zero the gradients
            self._weight_updated_in_prev_step = True
        else:
            self.zero_grad(state)

        if self.step_lr_interval == "step":
            self._update_lr_and_swa(state, self.train_progress.num_steps_completed)

        return total_grad_norm

    def get_next_train_batch(
        self, state: State, data_iter: Iterator[TData]
    ) -> Union[Iterator[TData], TData]:
        # Override the default behavior from PredictUnit in order to enable prefetching if possible.
        if self._train_step_requires_iterator:
            return data_iter
        return self._get_next_batch(state, data_iter)

    def get_next_eval_batch(
        self, state: State, data_iter: Iterator[TData]
    ) -> Union[Iterator[TData], TData]:
        # Override the default behavior from PredictUnit in order to enable prefetching if possible.
        if self._eval_step_requires_iterator:
            return data_iter
        return self._get_next_batch(state, data_iter)

    def get_next_predict_batch(
        self, state: State, data_iter: Iterator[TData]
    ) -> Union[Iterator[TData], TData]:
        # Override the default behavior from PredictUnit in order to enable prefetching if possible.
        if self._predict_step_requires_iterator:
            return data_iter
        return self._get_next_batch(state, data_iter)

    def _should_update_swa(self) -> bool:
        if not self.swa_params:
            return False

        swa_params = none_throws(self.swa_params)
        if self.step_lr_interval == "step":
            current_progress = self.train_progress.num_steps_completed
        else:
            # since num_epochs_completed is incremented prior to updating swa
            current_progress = self.train_progress.num_epochs_completed - 1

        if current_progress >= swa_params.warmup_steps_or_epochs:
            progress_since = current_progress - swa_params.warmup_steps_or_epochs
            update_freq = swa_params.step_or_epoch_update_freq
            return progress_since % update_freq == 0
        return False

    def _update_swa(self, state: State) -> None:
        with get_timing_context(
            state, f"{self.__class__.__name__}.stochastic_weight_avg_update"
        ):
            none_throws(self.swa_model).update_parameters(self.module)

    def _update_lr_and_swa(self, state: State, number_of_steps_or_epochs: int) -> None:
        if self._should_update_swa():
            self._update_swa(state)

        if self.swa_scheduler and (
            number_of_steps_or_epochs
            >= none_throws(self.swa_params).warmup_steps_or_epochs
        ):
            with get_timing_context(
                state, f"{self.__class__.__name__}.swa_lr_scheduler_step"
            ):
                none_throws(self.swa_scheduler).step()
        else:
            # optionally step lr scheduler if SWA not in use
            if self.lr_scheduler:
                with get_timing_context(
                    state, f"{self.__class__.__name__}.lr_scheduler_step"
                ):
                    self.step_lr_scheduler()
