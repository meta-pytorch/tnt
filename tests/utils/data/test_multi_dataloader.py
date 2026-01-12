#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
import unittest
from collections import Counter
from typing import Any, cast, Dict, Iterable, Iterator, List, Mapping, Union

import torch
from torch.utils.data import DataLoader, Dataset
from torchtnt.framework._test_utils import (
    generate_random_dataloader,
    generate_tensor_dataloader,
)
from torchtnt.utils.data.iterators import (
    AllDatasetBatches,
    DataIterationStrategy,
    InOrder,
    InOrderIterator,
    MultiIterator,
    RandomizedBatchSampler,
    RoundRobin,
    RoundRobinIterator,
    StoppingMechanism,
)
from torchtnt.utils.data.multi_dataloader import MultiDataLoader


class RandomDataset(Dataset[torch.Tensor]):
    def __init__(self, size: int, length: int) -> None:
        self.len: int = length
        self.data: torch.Tensor = torch.randn(length, size)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.len


class CustomRandomIterator(MultiIterator):
    """A test iterator.

    This returns batch from a random selected dataloader, until
    all dataloaders are exhausted.
    """

    def __init__(
        self,
        individual_dataloaders: Dict[str, DataLoader],
        iteration_strategy: DataIterationStrategy,
    ) -> None:
        super().__init__(individual_dataloaders, iteration_strategy)
        self.individual_iterators: Mapping[str, Any] = {
            name: iter(dl) for name, dl in individual_dataloaders.items()
        }
        self.remaining_dataloaders: List[str] = list(self.individual_iterators.keys())

    def __next__(self) -> Dict[str, Any]:
        if len(self.remaining_dataloaders) == 0:
            raise StopIteration

        cur_dataloader = random.choice(self.remaining_dataloaders)
        try:
            return {cur_dataloader: next(self.individual_iterators[cur_dataloader])}
        except StopIteration:
            self.remaining_dataloaders.remove(cur_dataloader)

            if len(self.remaining_dataloaders) == 0:
                raise StopIteration

            return self.__next__()


class TestMultiDataLoader(unittest.TestCase):
    def test_round_robin_smallest_dataset_exhausted(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(32, 8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(32, 16), batch_size=8)

        round_robin = RoundRobin(
            stopping_mechanism=StoppingMechanism.SMALLEST_DATASET_EXHAUSTED
        )
        multi_dataloader = iter(
            MultiDataLoader(
                self._get_dataloaders_dict(dataloader_1, dataloader_2),
                round_robin,
            )
        )

        # default iteration order
        iteration_order = ["1", "2"]

        # Since first dataset is exhausted in the first iteration,
        # it goes through only one full cycle.
        for index in range(2):
            batch = next(multi_dataloader)
            self.assertTrue(len(batch) == 1)
            self.assertTrue(iteration_order[index % 2] in batch)

        # Raises StopIteration after the first cycle
        with self.assertRaises(StopIteration):
            batch = next(multi_dataloader)

    def test_round_robin_all_datasets_exhausted(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(32, 8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(32, 16), batch_size=8)

        round_robin = RoundRobin(iteration_order=["2", "1"])
        multi_dataloader = iter(
            MultiDataLoader(
                self._get_dataloaders_dict(dataloader_1, dataloader_2),
                round_robin,
            )
        )

        # iteration order as defined in the strategy
        iteration_order = ["2", "1"]

        # Fetches 3 batches in total in the order ["2", "1", "2"]
        for index in range(3):
            batch = next(multi_dataloader)
            self.assertTrue(len(batch) == 1)
            self.assertTrue(iteration_order[index % 2] in batch)

        # StopIteration after both datasets have been exhausted
        with self.assertRaises(StopIteration):
            batch = next(multi_dataloader)

    def test_all_dataset_batches_all_datasets_exhausted(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(32, 8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(32, 16), batch_size=8)

        all_dataset_batches = AllDatasetBatches()

        multi_dataloader = iter(
            MultiDataLoader(
                self._get_dataloaders_dict(dataloader_1, dataloader_2),
                all_dataset_batches,
            )
        )

        # batch should contain both "1" and "2" keys
        batch = next(multi_dataloader)
        self.assertTrue(len(batch) == 2)
        self.assertTrue(("1" in batch) and ("2" in batch))

        # Dataset "1" was finished. batch now contains only samples from "2"
        batch = next(multi_dataloader)
        self.assertTrue(len(batch) == 1)
        self.assertTrue("2" in batch)

        # StopIteration after both datasets have been exhausted
        with self.assertRaises(StopIteration):
            batch = next(multi_dataloader)

    def test_all_dataset_batches_restart(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(32, 8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(32, 16), batch_size=8)

        all_dataset_batches = AllDatasetBatches(
            StoppingMechanism.RESTART_UNTIL_ALL_DATASETS_EXHAUSTED
        )

        multi_dataloader = iter(
            MultiDataLoader(
                self._get_dataloaders_dict(dataloader_1, dataloader_2),
                all_dataset_batches,
            )
        )

        # batch should contain both "1" and "2" keys
        batch = next(multi_dataloader)
        self.assertTrue(len(batch) == 2)
        self.assertTrue(("1" in batch) and ("2" in batch))

        # Dataset "1" was finished. batch now contains only samples from "2"
        batch = next(multi_dataloader)
        self.assertTrue(len(batch) == 2)
        self.assertTrue(("1" in batch) and ("2" in batch))

        # StopIteration after both datasets have been exhausted
        with self.assertRaises(StopIteration):
            batch = next(multi_dataloader)

    def test_all_dataset_batches_smallest_dataset_exhausted(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(32, 8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(32, 16), batch_size=8)

        all_dataset_batches = AllDatasetBatches(
            StoppingMechanism.SMALLEST_DATASET_EXHAUSTED
        )
        multi_dataloader = iter(
            MultiDataLoader(
                self._get_dataloaders_dict(dataloader_1, dataloader_2),
                all_dataset_batches,
            )
        )

        # batch should contain both "1" and "2" keys
        batch = next(multi_dataloader)
        self.assertTrue(len(batch) == 2)
        self.assertTrue(("1" in batch) and ("2" in batch))

        # StopIteration should be raised since the dataset "1" has no more samples
        with self.assertRaises(StopIteration):
            batch = next(multi_dataloader)

    def test_custom_iterator(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(32, 8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(32, 16), batch_size=8)

        multi_dataloader = iter(
            MultiDataLoader(
                self._get_dataloaders_dict(dataloader_1, dataloader_2),
                DataIterationStrategy(),
                CustomRandomIterator,
            )
        )

        # The two dataloaders constitute 3 batches in total
        for _ in range(3):
            batch = next(multi_dataloader)
            self.assertTrue(len(batch) == 1)

        # StopIteration after both datasets have been exhausted
        with self.assertRaises(StopIteration):
            batch = next(multi_dataloader)

    def test_random_sampling_dataloader_wrap_around(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(size=32, length=8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(size=32, length=16), batch_size=8)

        multi_dataloader = iter(
            MultiDataLoader(
                self._get_dataloaders_dict(dataloader_1, dataloader_2),
                RandomizedBatchSampler(
                    weights={"1": 1, "2": 100},
                    stopping_mechanism=StoppingMechanism.WRAP_AROUND_UNTIL_KILLED,
                ),
            )
        )

        # The two dataloaders constitute 3 batches in total
        selected_datasets = [next(multi_dataloader).popitem()[0] for _ in range(25)]
        # check that 2 is significantly more common
        counts = Counter(selected_datasets)
        self.assertTrue(counts["2"] > 0.8 * len(selected_datasets))

    def test_random_sampling_dataloader_with_empty_data(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(size=32, length=0), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(size=32, length=64), batch_size=8)

        multi_dataloader = iter(
            MultiDataLoader(
                self._get_dataloaders_dict(dataloader_1, dataloader_2),
                RandomizedBatchSampler(
                    weights={"1": 1, "2": 100},
                    stopping_mechanism=StoppingMechanism.ALL_DATASETS_EXHAUSTED,
                ),
                ignore_empty_data=True,
            )
        )

        # The first dataloader is empty
        # It will constitute 25 batches in total
        selected_datasets = [next(multi_dataloader).popitem()[0] for _ in range(8)]
        # check that 2 is the only one
        counts = Counter(selected_datasets)
        self.assertEqual(counts["2"], len(selected_datasets))

        # Now ensure that, without the `ignore_empty_data` flag, an exception is raised.
        with self.assertRaises(ValueError):
            MultiDataLoader(
                self._get_dataloaders_dict(dataloader_1, dataloader_2),
                RandomizedBatchSampler(weights={"1": 1, "2": 100}),
            )

    def test_random_sampling_dataloader_smallest_dataset_exhausted(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(size=32, length=8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(size=32, length=16), batch_size=8)

        multi_dataloader = MultiDataLoader(
            self._get_dataloaders_dict(dataloader_1, dataloader_2),
            RandomizedBatchSampler(
                weights={"1": 1, "2": 100},
                stopping_mechanism=StoppingMechanism.SMALLEST_DATASET_EXHAUSTED,
            ),
        )

        num_trials = 25
        len_counts = Counter()
        key_counts = Counter()
        for _ in range(num_trials):
            batches = list(iter(multi_dataloader))
            len_counts[len(batches)] += 1
            key_counts.update(key for batch in batches for key in batch.keys())
        # Most trials should only return two batches
        self.assertTrue(len_counts[2] > 0.8 * num_trials)
        # Most batches overall should be from dataset "2"
        self.assertTrue(key_counts["2"] > 0.8 * sum(key_counts.values()))

    def test_random_sampling_dataloader_all_datasets_exhausted(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(size=32, length=8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(size=32, length=16), batch_size=8)

        multi_dataloader = MultiDataLoader(
            self._get_dataloaders_dict(dataloader_1, dataloader_2),
            RandomizedBatchSampler(
                weights={"1": 1, "2": 100},
            ),
        )

        num_trials = 25
        trials = [list(iter(multi_dataloader)) for _ in range(num_trials)]

        # All trials should return three batches
        self.assertTrue(all(len(data) == 3 for data in trials))

        # Most trials should start with dataset "2"
        key_counts = Counter(data[0].popitem()[0] for data in trials)
        self.assertTrue(key_counts["2"] > 0.8 * num_trials)

    def test_random_sampling_dataloader_restart_until_all_datasets_exhausted(
        self,
    ) -> None:
        dataloader_1 = DataLoader(RandomDataset(size=32, length=8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(size=32, length=256), batch_size=8)

        multi_dataloader = iter(
            MultiDataLoader(
                self._get_dataloaders_dict(dataloader_1, dataloader_2),
                RandomizedBatchSampler(
                    weights={"1": 100, "2": 1},
                    stopping_mechanism=StoppingMechanism.RESTART_UNTIL_ALL_DATASETS_EXHAUSTED,
                ),
            )
        )

        num_trials = 32
        selected_datasets = [
            next(multi_dataloader).popitem()[0] for _ in range(num_trials)
        ]
        # check that 1 is significantly more common, even if it is shorter
        counts = Counter(selected_datasets)
        self.assertTrue(counts["1"] > 0.8 * len(selected_datasets))

    def test_inorder(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(32, 8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(32, 16), batch_size=8)

        in_order = InOrder()

        multi_dataloader = iter(
            MultiDataLoader(
                self._get_dataloaders_dict(dataloader_1, dataloader_2),
                in_order,
            )
        )

        batch = next(multi_dataloader)
        self.assertTrue(len(batch) == 1)
        self.assertTrue("1" in batch)

        for _ in range(2):
            batch = next(multi_dataloader)
            self.assertTrue(len(batch) == 1)
            self.assertTrue("2" in batch)

        # Raises StopIteration after all exhausted
        with self.assertRaises(StopIteration):
            batch = next(multi_dataloader)

    def test_in_order_with_repetitions(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(32, 8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(32, 16), batch_size=8)

        in_order = InOrder(iteration_order=["2", "1", "2"])

        multi_dataloader = iter(
            MultiDataLoader(
                self._get_dataloaders_dict(dataloader_1, dataloader_2),
                in_order,
            )
        )

        # first 2 batches from dataset 2
        for _ in range(2):
            batch = next(multi_dataloader)
            self.assertTrue(len(batch) == 1)
            self.assertTrue("2" in batch)

        # next batch from dataset 1
        batch = next(multi_dataloader)
        self.assertTrue(len(batch) == 1)
        self.assertTrue("1" in batch)

        # last 2 batches from dataset 2
        for _ in range(2):
            batch = next(multi_dataloader)
            self.assertTrue(len(batch) == 1)
            self.assertTrue("2" in batch)

        # Raises StopIteration after all exhausted
        with self.assertRaises(StopIteration):
            batch = next(multi_dataloader)

    def test_state_dict_load_state_dict(self) -> None:
        class DummyIterable:
            def __init__(self, vals: List[int]) -> None:
                self.vals = vals
                # Start at -1 since an iterator is generated when the MultiDataLoader is constructed
                # while checking for missing data
                self.iter_count = -1

            def __iter__(self) -> Iterator[int]:
                self.iter_count += 1
                return iter(self.vals)

            def state_dict(self) -> Dict[str, Any]:
                return {"iter_count": self.iter_count}

            def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
                self.iter_count = state_dict["iter_count"]

        iterable_1 = DummyIterable([1, 2, 3])
        iterable_2 = DummyIterable([4, 5, 6])
        # Add an iterable which does not implement the stateful protocol
        iterable_3 = [7, 8, 9]

        multi_dataloader = MultiDataLoader(
            {"foo": iterable_1, "bar": iterable_2, "baz": iterable_3}, InOrder()
        )

        # Generate state dict from initial state
        original_state_dict = multi_dataloader.state_dict()

        # Confirm keys are appropriately set
        self.assertIn("foo", original_state_dict)
        self.assertIn("iter_count", original_state_dict["foo"])
        self.assertEqual(0, original_state_dict["foo"]["iter_count"])
        self.assertIn("bar", original_state_dict)
        self.assertIn("iter_count", original_state_dict["bar"])
        self.assertEqual(0, original_state_dict["bar"]["iter_count"])
        self.assertNotIn("baz", original_state_dict)

        for _ in multi_dataloader:
            pass

        foo_iterable = cast(
            DummyIterable, multi_dataloader.individual_dataloaders["foo"]
        )
        bar_iterable = cast(
            DummyIterable, multi_dataloader.individual_dataloaders["bar"]
        )
        self.assertEqual(foo_iterable.iter_count, 1)
        self.assertEqual(bar_iterable.iter_count, 1)

        new_state_dict = multi_dataloader.state_dict()

        # Load state dict to reset to initial state
        multi_dataloader.load_state_dict(original_state_dict)
        self.assertEqual(foo_iterable.iter_count, 0)
        self.assertEqual(bar_iterable.iter_count, 0)

        # instantiate a new multi-dataloader with a new different name
        new_multi_dataloader = MultiDataLoader(
            {
                "foo": DummyIterable([1, 2, 3]),
                "qux": DummyIterable([4, 5, 6]),
                "baz": [7, 8, 9],
            },
            InOrder(),
        )
        new_multi_dataloader.load_state_dict(new_state_dict)
        # foo's count should be loaded correctly
        self.assertEqual(
            cast(
                DummyIterable, new_multi_dataloader.individual_dataloaders["foo"]
            ).iter_count,
            1,
        )
        # qux's iter_count should still be 0 because it was not in the original state dict
        self.assertEqual(
            cast(
                DummyIterable, new_multi_dataloader.individual_dataloaders["qux"]
            ).iter_count,
            0,
        )

    def test_multi_dataloader_state_dict_with_iterator_state(self) -> None:
        dataloader_1 = generate_random_dataloader(
            num_samples=8, input_dim=1, batch_size=8
        )
        dataloader_2 = generate_random_dataloader(
            num_samples=16, input_dim=1, batch_size=8
        )
        multi_dataloader = MultiDataLoader(
            self._get_dataloaders_dict(dataloader_1, dataloader_2),
            RoundRobin(),
        )

        multi_dl_state_dict = multi_dataloader.state_dict()
        # before creating the iterator we don't expect the iterator_state to be present in the dl state dict
        self.assertFalse("iterator_state" in multi_dl_state_dict)

        multi_dl_iter = iter(multi_dataloader)
        multi_dl_state_dict = multi_dataloader.state_dict()
        self.assertTrue("iterator_state" in multi_dl_state_dict)
        self.assertEqual(
            multi_dl_state_dict["iterator_state"],
            {"cur_dataloader": "1", "finished_dataloaders": []},
        )
        next(multi_dl_iter)  # should return batch from 1
        next(multi_dl_iter)  # should return batch from 2
        next(
            multi_dl_iter
        )  # should return batch from 2 after raising StopIteration from the first dl
        multi_dl_state_dict = multi_dataloader.state_dict()
        self.assertTrue("iterator_state" in multi_dl_state_dict)
        self.assertEqual(
            multi_dl_state_dict["iterator_state"],
            {"cur_dataloader": "2", "finished_dataloaders": ["1"]},
        )

        # create fresh dl and load state dict. assert that the initial values are updated.
        multi_dataloader_2 = MultiDataLoader(
            self._get_dataloaders_dict(dataloader_1, dataloader_2),
            RoundRobin(),
        )
        multi_dataloader_2.load_state_dict(multi_dl_state_dict)
        round_robin_iter = cast(RoundRobinIterator, iter(multi_dataloader_2))
        self.assertEqual(round_robin_iter.cur_dataloader, "2")
        self.assertEqual(round_robin_iter.finished_dataloaders, ["1"])

        # verify that after calling iter() again, values are reset
        round_robin_iter = cast(RoundRobinIterator, iter(multi_dataloader_2))
        self.assertEqual(round_robin_iter.cur_dataloader, "1")
        self.assertEqual(round_robin_iter.finished_dataloaders, [])

    def test_invalid_load_state_dict(self) -> None:
        dataloader_1 = generate_random_dataloader(
            num_samples=8, input_dim=1, batch_size=8
        )
        dataloader_2 = generate_random_dataloader(
            num_samples=16, input_dim=1, batch_size=8
        )
        multi_dataloader = MultiDataLoader(
            self._get_dataloaders_dict(dataloader_1, dataloader_2),
            RoundRobin(),
        )

        # invalid state dict - finished dataloaders and curr dataloader do not exist
        multi_dataloader.load_state_dict(
            {"finished_dataloaders": ["3"], "cur_dataloader": "4"}
        )
        round_robin_iter = cast(RoundRobinIterator, iter(multi_dataloader))
        # ensure the iterator state is not changed
        self.assertEqual(round_robin_iter.cur_dataloader, "1")
        self.assertEqual(round_robin_iter.finished_dataloaders, [])

    def test_state_dict_with_non_stateful_iterator(self) -> None:
        dataloader_1 = generate_random_dataloader(
            num_samples=8, input_dim=1, batch_size=8
        )
        dataloader_2 = generate_random_dataloader(
            num_samples=16, input_dim=1, batch_size=8
        )
        multi_dataloader = MultiDataLoader(
            self._get_dataloaders_dict(dataloader_1, dataloader_2),
            DataIterationStrategy(),
            CustomRandomIterator,
        )
        iter(multi_dataloader)
        self.assertFalse("iterator_state" in multi_dataloader.state_dict())

    def _get_dataloaders_dict(
        self, first_dataloader: DataLoader, second_dataloader: DataLoader
    ) -> Dict[str, Union[DataLoader, Iterable[object]]]:
        return {"1": first_dataloader, "2": second_dataloader}

    def test_state_dict_with_inorder_iterator(self) -> None:
        dataloader_1 = generate_tensor_dataloader(torch.tensor([1, 2]), batch_size=1)
        dataloader_2 = generate_tensor_dataloader(torch.tensor([3, 4]), batch_size=1)
        dataloader_3 = generate_tensor_dataloader(torch.tensor([5, 6]), batch_size=1)

        dataloaders_dict: Dict[str, Union[DataLoader, Iterable[object]]] = {
            "dataloader_1": dataloader_1,
            "dataloader_2": dataloader_2,
            "dataloader_3": dataloader_3,
        }

        multi_dataloader = MultiDataLoader(
            dataloaders_dict,
            InOrder(iteration_order=["dataloader_1", "dataloader_2", "dataloader_3"]),
        )

        mdl_iter = iter(multi_dataloader)

        # Exhaust first dataset
        self.assertEqual(next(mdl_iter)["dataloader_1"], [torch.tensor([1])])
        self.assertEqual(next(mdl_iter)["dataloader_1"], [torch.tensor([2])])

        # We expect same iterator since we haven't raise StopIteration
        mdl_sd = multi_dataloader.state_dict()
        self.assertEqual(
            mdl_sd["iterator_state"],
            {
                "iterators_finished": 0,
                "cur_iterator": "dataloader_1",
            },
        )

        # Start second dataset
        self.assertEqual(next(mdl_iter)["dataloader_2"], [torch.tensor([3])])

        mdl_sd = multi_dataloader.state_dict()
        self.assertEqual(
            mdl_sd["iterator_state"],
            {
                "iterators_finished": 1,
                "cur_iterator": "dataloader_2",
            },
        )

        # Create new dataloader and verify restore
        multi_dataloader_2 = MultiDataLoader(
            dataloaders_dict,
            InOrder(iteration_order=["dataloader_1", "dataloader_2", "dataloader_3"]),
        )
        multi_dataloader_2.load_state_dict(mdl_sd)
        in_order_iter = cast(InOrderIterator, iter(multi_dataloader_2))
        self.assertEqual(in_order_iter.cur_iterator, "dataloader_1")
        self.assertEqual(in_order_iter.iterators_finished, 1)

        # Calling next should update the currrent iterator
        # individual dl is not stateful
        self.assertEqual(next(in_order_iter)["dataloader_2"], [torch.tensor([3])])
        self.assertEqual(in_order_iter.cur_iterator, "dataloader_2")

        # verify that after calling iter(), everything is reset
        in_order_iter_2 = cast(InOrderIterator, iter(multi_dataloader_2))
        self.assertEqual(in_order_iter_2.cur_iterator, "dataloader_1")
        self.assertEqual(in_order_iter_2.iterators_finished, 0)
        self.assertEqual(next(in_order_iter_2)["dataloader_1"], [torch.tensor([1])])
