# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import deque
from typing import Callable, Iterator, List, Optional, Sequence, Union

import torch
from pyre_extensions import none_throws
from torch.autograd.graph import GradientEdge, Node
from torch.utils.hooks import RemovableHandle


def _get_grad_fn_or_grad_acc(t: Union[torch.Tensor, GradientEdge]) -> Node:
    if isinstance(t, torch.Tensor):
        return none_throws(t.grad_fn)
    else:
        # pyre-ignore Undefined attribute [16]: `GradientEdge` has no attribute `function`.
        return t.function if t is not None else None


def register_nan_hooks_on_whole_graph(  # noqa: C901
    t_outputs: Sequence[Union[torch.Tensor, GradientEdge]],
) -> Callable[[], None]:
    """
    Registers a nan hook on the whole graph of the given tensors. The hook will throw error if a nan is detected.

    This is useful if you want training to halt when a nan is detected during autograd process (ie loss is inf or nan).

    Usage:

        >>> class NaNFunction(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input):
                    return input.clone()

                @staticmethod
                def backward(ctx, grad_output):
                    return torch.tensor([float("nan")], device="cpu")
        >>> x = torch.tensor([1.0], device="cpu", requires_grad=True)
        >>> out = NaNFunction.apply(x)
        >>> _ = register_nan_hooks_on_whole_graph([out])
        >>> out.backward()
        RuntimeError: Detected NaN in 'grad_inputs[0]' after executing Node

    """

    grad_fns = list(map(_get_grad_fn_or_grad_acc, t_outputs))

    def iter_graph(roots: List[torch.autograd.graph.Node]) -> Iterator[Node]:
        if not roots:
            return
        seen = set()
        q = deque()
        for node in roots:
            if node is not None and node not in seen:
                seen.add(node)
                q.append(node)
        while q:
            node = q.popleft()
            for fn, _ in node.next_functions:
                if fn is None or fn in seen:
                    continue
                seen.add(fn)
                q.append(fn)
            yield node

    def _assert_no_nan_tensor(t: Optional[torch.Tensor], msg: str) -> None:
        if t is not None:
            torch._assert_async(torch.logical_not(torch.any(torch.isnan(t))), msg)

    def posthook(
        grad_inputs: Sequence[Optional[torch.Tensor]],
        grad_outputs: Sequence[Optional[torch.Tensor]],
    ) -> None:
        node = torch._C._current_autograd_node()
        for i, g_in in enumerate(grad_inputs):
            _assert_no_nan_tensor(
                g_in, f"Detected NaN in 'grad_inputs[{i}]' after executing Node: {node}"
            )

    handles: List[RemovableHandle] = []
    for node in iter_graph(grad_fns):
        posthandle = node.register_hook(posthook)
        handles.append(posthandle)

    def unregister_hooks() -> None:
        for handle in handles:
            handle.remove()

    return unregister_hooks


def check_for_nan_or_inf(
    tensor: torch.Tensor, msg: str = "Detected NaN or Inf in tensor"
) -> None:
    """
    Asynchronously assert that the tensor is neither NaN nor infinity. This will
    produce a cuda device side assert error if tensor on gpu.
    """

    torch._assert_async(
        torch.logical_not(torch.any(torch.isnan(tensor) | torch.isinf(tensor))),
        msg,
    )
