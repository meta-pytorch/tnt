#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import time
import unittest
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import cast, Deque, List

import torch
from torchtnt.utils.memory import (
    get_tensor_size_bytes_map,
    measure_rss_deltas,
    RSSProfiler,
)


class MemoryTest(unittest.TestCase):
    def test_get_tensor_size_bytes_map_with_tensor_input(self) -> None:
        """Test behavior with a Tensor input"""

        inputs = torch.rand(1, 2, 3)
        tensor_map = get_tensor_size_bytes_map(inputs)
        self.assertEqual(len(tensor_map), 1)
        self.assertTrue(inputs in tensor_map)
        self.assertEqual(
            tensor_map[inputs], inputs.size().numel() * inputs.element_size()
        )

    def test_get_tensor_size_bytes_map_with_named_tuple_input(self) -> None:
        """Test behavior with a named tuple input"""

        Object = namedtuple("Object", ["x", "y"])
        inputs = Object(x=torch.rand(1, 2, 3), y=torch.rand(4, 5, 6))
        tensor_map = get_tensor_size_bytes_map(inputs)
        self.assertEqual(len(tensor_map), 2)
        self.assertTrue(inputs.x in tensor_map)
        self.assertEqual(
            tensor_map[inputs.x], inputs.x.size().numel() * inputs.x.element_size()
        )
        self.assertTrue(inputs.y in tensor_map)
        self.assertEqual(
            tensor_map[inputs.y], inputs.y.size().numel() * inputs.y.element_size()
        )

    def test_get_tensor_size_bytes_map_with_mapping_input(self) -> None:
        """Test behavior with a Mapping input"""

        inputs = {"x": torch.rand(1, 2, 4), "y": torch.rand(5, 10, 7)}
        tensor_map = get_tensor_size_bytes_map(inputs)
        self.assertEqual(len(tensor_map), 2)
        self.assertTrue(inputs["x"] in tensor_map)
        self.assertEqual(
            tensor_map[inputs["x"]],
            inputs["x"].size().numel() * inputs["x"].element_size(),
        )
        self.assertTrue(inputs["y"] in tensor_map)
        self.assertEqual(
            tensor_map[inputs["y"]],
            inputs["y"].size().numel() * inputs["y"].element_size(),
        )

    def test_get_tensor_size_bytes_map_with_sequence_input(self) -> None:
        """Test behavior with a Sequence input"""

        inputs = [torch.rand(3, 2, 1), torch.rand(9, 8, 7)]
        tensor_map = get_tensor_size_bytes_map(inputs)
        self.assertEqual(len(tensor_map), 2)
        self.assertTrue(inputs[0] in tensor_map)
        self.assertEqual(
            tensor_map[inputs[0]], inputs[0].size().numel() * inputs[0].element_size()
        )
        self.assertTrue(inputs[1] in tensor_map)
        self.assertEqual(
            tensor_map[inputs[1]], inputs[1].size().numel() * inputs[1].element_size()
        )

    def test_get_tensor_size_bytes_map_with_input_attributes(self) -> None:
        """Test behavior with an input that has __dict__ attributes"""

        class Object:
            def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
                self.x = x
                self.y = y
                self.z = 1

        inputs = Object(x=torch.rand(2, 4, 6), y=torch.rand(8, 10, 12))
        tensor_map = get_tensor_size_bytes_map(inputs)
        self.assertEqual(len(tensor_map), 2)
        self.assertTrue(inputs.x in tensor_map)
        self.assertEqual(
            tensor_map[inputs.x], inputs.x.size().numel() * inputs.x.element_size()
        )
        self.assertTrue(inputs.y in tensor_map)
        self.assertEqual(
            tensor_map[inputs.y], inputs.y.size().numel() * inputs.y.element_size()
        )

    def test_get_tensor_size_bytes_map_with_dataclass_input(self) -> None:
        """Test behavior with a dataclass input"""

        @dataclass
        class TestTensorDataClass:
            x: torch.Tensor
            y: torch.Tensor

        inputs = TestTensorDataClass(x=torch.rand(6, 4, 2), y=torch.rand(8, 6, 4))
        tensor_map = get_tensor_size_bytes_map(inputs)
        self.assertEqual(len(tensor_map), 2)
        self.assertTrue(inputs.x in tensor_map)
        self.assertEqual(
            tensor_map[inputs.x], inputs.x.size().numel() * inputs.x.element_size()
        )
        self.assertTrue(inputs.y in tensor_map)
        self.assertEqual(
            tensor_map[inputs.y], inputs.y.size().numel() * inputs.y.element_size()
        )

    def test_get_tensor_size_bytes_map_with_nested_input(self) -> None:
        """Test behavior with a nested input"""

        inputs = {
            "x": torch.rand(1, 2, 3),
            "y": {"z": torch.rand(4, 5, 6), "t": torch.rand(7, 8, 9)},
        }
        tensor_map = get_tensor_size_bytes_map(inputs)
        self.assertEqual(len(tensor_map), 3)
        self.assertTrue(inputs["x"] in tensor_map)
        input_x = cast(torch.Tensor, inputs["x"])
        tensor_map_x = cast(torch.Tensor, tensor_map[input_x])
        self.assertEqual(
            tensor_map_x,
            input_x.size().numel() * input_x.element_size(),
        )
        # pyre-fixme[6]: For 1st argument expected `Union[None, List[typing.Any],
        #  int, slice, Tensor, typing.Tuple[typing.Any, ...]]` but got `str`.
        self.assertTrue(inputs["y"]["z"] in tensor_map)
        self.assertEqual(
            # pyre-fixme[6]: For 1st argument expected `Union[None,
            #  List[typing.Any], int, slice, Tensor, typing.Tuple[typing.Any, ...]]`
            #  but got `str`.
            tensor_map[inputs["y"]["z"]],
            # pyre-fixme[6]: For 1st argument expected `Union[None,
            #  List[typing.Any], int, slice, Tensor, typing.Tuple[typing.Any, ...]]`
            #  but got `str`.
            inputs["y"]["z"].size().numel() * inputs["y"]["z"].element_size(),
        )
        # pyre-fixme[6]: For 1st argument expected `Union[None, List[typing.Any],
        #  int, slice, Tensor, typing.Tuple[typing.Any, ...]]` but got `str`.
        self.assertTrue(inputs["y"]["t"] in tensor_map)
        self.assertEqual(
            # pyre-fixme[6]: For 1st argument expected `Union[None,
            #  List[typing.Any], int, slice, Tensor, typing.Tuple[typing.Any, ...]]`
            #  but got `str`.
            tensor_map[inputs["y"]["t"]],
            # pyre-fixme[6]: For 1st argument expected `Union[None,
            #  List[typing.Any], int, slice, Tensor, typing.Tuple[typing.Any, ...]]`
            #  but got `str`.
            inputs["y"]["t"].size().numel() * inputs["y"]["t"].element_size(),
        )

    def test_get_tensor_size_bytes_map_with_metric_input(self) -> None:
        """Test behavior with a Metric input"""

        class WindowBuffer:
            def __init__(self, max_size: int, max_buffer_count: int) -> None:
                self._max_size: int = max_size
                self._max_buffer_count: int = max_buffer_count

                self._buffers: Deque[torch.Tensor] = deque(maxlen=max_buffer_count)
                self._used_sizes: Deque[int] = deque(maxlen=max_buffer_count)
                self._window_used_size = 0

            def aggregate_state(
                self, window_state: torch.Tensor, curr_state: torch.Tensor, size: int
            ) -> None:
                def remove(window_state: torch.Tensor) -> None:
                    window_state -= self._buffers.popleft()
                    self._window_used_size -= self._used_sizes.popleft()

                if len(self._buffers) == self._buffers.maxlen:
                    remove(window_state)

                self._buffers.append(curr_state)
                self._used_sizes.append(size)
                window_state += curr_state
                self._window_used_size += size

                while self._window_used_size > self._max_size:
                    remove(window_state)

            @property
            def buffers(self) -> Deque[torch.Tensor]:
                return self._buffers

        class RandomModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.window_buffer = WindowBuffer(2000, 1000)
                self.window_buffer.aggregate_state(
                    torch.rand(6, 7, 8), torch.rand(6, 7, 8), 5
                )
                self.x = torch.rand(1, 2)
                self.y = [torch.rand(4, 5), torch.rand(9, 10)]

        class RandomModuleList(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.metric_list = torch.nn.ModuleList([RandomModule(), RandomModule()])

        inputs = RandomModuleList()
        tensor_map = get_tensor_size_bytes_map(inputs)
        self.assertEqual(
            # pyrefly: ignore [missing-attribute]
            len(tensor_map),
            # pyrefly: ignore [missing-attribute]
            2 * len(inputs.metric_list[0].window_buffer.buffers) + 6,
        )
        for metric in inputs.metric_list:
            metric = cast(RandomModule, metric)
            self.assertIn(metric.x, tensor_map)
            self.assertEqual(
                tensor_map[metric.x], metric.x.size().numel() * metric.x.element_size()
            )
            self.assertIn(metric.y[0], tensor_map)
            self.assertEqual(
                tensor_map[metric.y[0]],
                metric.y[0].size().numel() * metric.y[0].element_size(),
            )
            self.assertIn(metric.y[1], tensor_map)
            self.assertEqual(
                tensor_map[metric.y[1]],
                metric.y[1].size().numel() * metric.y[1].element_size(),
            )
            while metric.window_buffer.buffers:
                z = metric.window_buffer.buffers.popleft()
                self.assertTrue(z in tensor_map)
                self.assertEqual(tensor_map[z], z.size().numel() * z.element_size())

    def test_rss_measure_func(self) -> None:
        rss_deltas = []
        with measure_rss_deltas(rss_deltas=rss_deltas):
            torch.randn(5000, 5000)
            time.sleep(2)
        self.assertTrue(len(rss_deltas) > 0)
        self.assertTrue(_has_non_zero_elem(rss_deltas))

    def test_rss_profiler(self) -> None:
        rss_profiler = RSSProfiler()
        with rss_profiler.profile("foo"):
            torch.randn(5000, 5000)
            time.sleep(2)
        self.assertTrue(len(rss_profiler.rss_deltas_bytes["foo"]) > 0)
        self.assertTrue(_has_non_zero_elem(rss_profiler.rss_deltas_bytes["foo"]))


def _has_non_zero_elem(rss_deltas: List[int]) -> bool:
    for rss in rss_deltas:
        if rss > 0:
            return True

    return False
