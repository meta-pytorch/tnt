#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
import random
from abc import abstractmethod
from dataclasses import dataclass
from itertools import cycle
from typing import (
    Any,
    cast,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Type,
    TYPE_CHECKING,
    Union,
)

from python.migrations.py310 import StrEnum310

try:
    from enum import StrEnum
except ImportError:

    class StrEnum(StrEnum310):
        pass


import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class DataIterationStrategy:
    pass


class MultiIterator(Iterator[Dict[str, Any]]):
    """MultiIterator defines the iteration logic to get a batch, given
    batches from all individual dataloaders.
    iteration_strategy can include accompanying parameters for a particular
    iterator, like cycling order for the dataloaders.

    Args:
        individual_dataloaders (Mapping[str, Union[DataLoader, Iterable]]): A mapping of DataLoaders or Iterables with dataloader name as key
            and dataloader/iterable object as value.
        iteration_strategy (DataIterationStrategy): A dataclass indicating how the dataloaders are iterated over.

    Note:
        TorchData (https://pytorch.org/data/beta/index.html) also has generic multi-data
            sources reading support to achieve the same functionality provided by MultiIterator.
        For example, `mux`, `mux_longest`, `cycle`, `zip` etc. Please refer to the documentation for more details.

    """

    def __init__(
        self,
        individual_dataloaders: Mapping[str, Union[DataLoader, Iterable[object]]],
        iteration_strategy: DataIterationStrategy,
    ) -> None:
        self.individual_dataloaders = individual_dataloaders
        self.iteration_strategy = iteration_strategy

    def __str__(self) -> str:
        return str(self.iteration_strategy)

    @abstractmethod
    def __next__(self) -> Dict[str, Any]:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


# pyrefly: ignore [invalid-inheritance]
class StoppingMechanism(StrEnum):
    ALL_DATASETS_EXHAUSTED = "ALL_DATASETS_EXHAUSTED"
    SMALLEST_DATASET_EXHAUSTED = "SMALLEST_DATASET_EXHAUSTED"
    RESTART_UNTIL_ALL_DATASETS_EXHAUSTED = "RESTART_UNTIL_ALL_DATASETS_EXHAUSTED"

    # used with RandomizedBatchSampler
    WRAP_AROUND_UNTIL_KILLED = "WRAP_AROUND_UNTIL_KILLED"


@dataclass
class RoundRobin(DataIterationStrategy):
    stopping_mechanism: StoppingMechanism = StoppingMechanism.ALL_DATASETS_EXHAUSTED
    iteration_order: Optional[List[str]] = None


class RoundRobinIterator(MultiIterator):
    """RoundRobinIterator cycles over the dataloader one by one.
    Iterating order can be defined via RobinRobin strategy.

    This supports two stopping mechanisms:
    1. ALL_DATASETS_EXHAUSTED: Iterates till the largest dataset is exhausted,
    while skipping those that are done
    2. SMALLEST_DATASET_EXHAUSTED: Stops iteration once the smallest dataset
    has been exhausted

    Returns batches of the format: {dataloader_name: batch_from_dataloader}

    Args:
        individual_dataloaders (Mapping[str, Union[DataLoader, Iterable]]): A mapping of DataLoaders or Iterables with dataloader name as key
        and dataloader/iterable object as value.
        iteration_strategy (RoundRobin): A RoundRobin dataclass indicating how the dataloaders are iterated over.

    Examples:
        >>> loaders = {'a': torch.utils.data.DataLoader(range(4), batch_size=4),
            'b': torch.utils.data.DataLoader(range(15), batch_size=5)}
        >>> round_robin_strategy = RoundRobin(
                stopping_mechanism=StoppingMechanism.ALL_DATASETS_EXHAUSTED
            )
        >>> combined_iterator = RoundRobinIterator(loaders, round_robin_strategy)
        >>> for item in combined_iterator:
                print(item)
        {'a': tensor([0, 1, 2, 3])}
        {'b': tensor([0, 1, 2, 3, 4])}
        {'b': tensor([5, 6, 7, 8, 9])}
        {'b': tensor([10, 11, 12, 13, 14])}

    """

    def __init__(
        self,
        individual_dataloaders: Mapping[str, Union[DataLoader, Iterable[object]]],
        iteration_strategy: RoundRobin,
    ) -> None:
        super().__init__(individual_dataloaders, iteration_strategy)
        self.iteration_strategy = iteration_strategy

        if (
            self.iteration_strategy.stopping_mechanism
            == StoppingMechanism.WRAP_AROUND_UNTIL_KILLED
        ):
            raise NotImplementedError(
                "WRAP_AROUND_UNTIL_KILLED is not implemented for RoundRobin"
            )
        self.individual_iterators: Mapping[
            str, Union[Iterator[DataLoader], Iterator[object]]
        ] = {name: iter(dl) for name, dl in individual_dataloaders.items()}
        round_robin_order = iteration_strategy.iteration_order or list(
            self.individual_iterators.keys()
        )
        self.dataloader_cycle: Iterator[str] = cycle(round_robin_order)
        self.cur_dataloader: str = round_robin_order[0]
        self.finished_dataloaders: List[str] = []

    def __next__(self) -> Dict[str, Any]:
        if len(self.finished_dataloaders) == len(self.individual_iterators):
            raise StopIteration

        self.cur_dataloader = next(self.dataloader_cycle)
        while self.cur_dataloader in self.finished_dataloaders:
            self.cur_dataloader = next(self.dataloader_cycle)
        try:
            return {
                self.cur_dataloader: next(
                    self.individual_iterators[self.cur_dataloader]
                )
            }
        except StopIteration:
            if (
                # pyrefly: ignore [missing-attribute]
                self.iteration_strategy.stopping_mechanism
                == StoppingMechanism.SMALLEST_DATASET_EXHAUSTED
            ):
                raise StopIteration

            self.finished_dataloaders.append(self.cur_dataloader)

            if len(self.finished_dataloaders) == len(self.individual_iterators):
                raise StopIteration

            return self.__next__()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "finished_dataloaders": self.finished_dataloaders,
            "cur_dataloader": self.cur_dataloader,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        logger.info(
            f"Loading RoundRobinIterator state. Finished dataloaders: {state_dict['finished_dataloaders']} and trying to set cur_dataloader to {self.cur_dataloader}"
        )
        self.finished_dataloaders = state_dict["finished_dataloaders"]
        cur_dataloader = state_dict["cur_dataloader"]
        if cur_dataloader not in self.dataloader_cycle:
            logger.warning(
                f"Did not find {cur_dataloader} in {list(self.dataloader_cycle)}. Skipping setting cur_dataloader"
            )
            return
        while self.cur_dataloader != cur_dataloader:
            self.cur_dataloader = next(self.dataloader_cycle)


@dataclass
class AllDatasetBatches(DataIterationStrategy):
    stopping_mechanism: StoppingMechanism = StoppingMechanism.ALL_DATASETS_EXHAUSTED


class AllDatasetBatchesIterator(MultiIterator):
    """AllDatasetBatchesIterator returns a dict containing batches from all dataloaders.
    When the stopping mechanism is set to ALL_DATASETS_EXHAUSTED, it will skip over the
    finished datasets.


    This supports three stopping mechanisms:
    1. `ALL_DATASETS_EXHAUSTED`: Iterates till the largest dataset is exhausted,
    while skipping those that are done
    2. `SMALLEST_DATASET_EXHAUSTED`: Stops iteration once the smallest dataset
    has been exhausted
    3. `RESTART_UNTIL_ALL_DATASETS_EXHAUSTED`: Iterates until the largest dataset
    is exhausted, while restarting those that are done

    Returns batches of the format: {
        dataloader_1_name: batch_obtained_from_dataloader_1,
        dataloader_2_name: batch_obtained_from_dataloader_2,
    }

    Args:
        individual_dataloaders (Mapping[str, Union[DataLoader, Iterable]]): A mapping of DataLoaders or Iterables with dataloader name as key
        and dataloader/iterable object as value.
        iteration_strategy (AllDatasetBatches): A AllDatasetBatches dataclass indicating how the dataloaders are iterated over.

    Examples:
        >>> loaders = {'a': torch.utils.data.DataLoader(range(4), batch_size=4),
            'b': torch.utils.data.DataLoader(range(15), batch_size=5)}
        >>> all_dataset_batch_strategy = AllDatasetBatches(
                stopping_mechanism=StoppingMechanism.ALL_DATASETS_EXHAUSTED
            )
        >>> combined_iterator = AllDatasetBatchesIterator(loaders, all_dataset_batch_strategy)
        >>> for item in combined_iterator:
                print(item)
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([0, 1, 2, 3, 4])}
        {'b': tensor([5, 6, 7, 8, 9])}
        {'b': tensor([10, 11, 12, 13, 14])}

    """

    def __init__(
        self,
        individual_dataloaders: Mapping[str, Union[DataLoader, Iterable[object]]],
        iteration_strategy: AllDatasetBatches,
    ) -> None:
        super().__init__(individual_dataloaders, iteration_strategy)
        self.iteration_strategy = iteration_strategy
        if (
            self.iteration_strategy.stopping_mechanism
            == StoppingMechanism.WRAP_AROUND_UNTIL_KILLED
        ):
            raise NotImplementedError(
                "WRAP_AROUND_UNTIL_KILLED is not implemented for AllDatasetBatches"
            )
        self.individual_iterators: Dict[
            str, Union[Iterator[DataLoader], Iterator[object]]
        ] = {name: iter(dl) for name, dl in individual_dataloaders.items()}
        self.iterators_finished: List[str] = []

    def __next__(self) -> Dict[str, Any]:
        batch_dict = {}
        for iterator in self.individual_iterators:
            try:
                # pyrefly: ignore [unsupported-operation]
                batch_dict[iterator] = next(self.individual_iterators[iterator])
            except StopIteration:
                if (
                    # pyrefly: ignore [missing-attribute]
                    self.iteration_strategy.stopping_mechanism
                    == StoppingMechanism.SMALLEST_DATASET_EXHAUSTED
                ):
                    raise StopIteration

                elif (
                    self.iteration_strategy.stopping_mechanism
                    == StoppingMechanism.RESTART_UNTIL_ALL_DATASETS_EXHAUSTED
                ):
                    if iterator not in self.iterators_finished:
                        self.iterators_finished.append(iterator)
                    if len(self.iterators_finished) == len(self.individual_iterators):
                        raise StopIteration
                    else:
                        self.individual_iterators[iterator] = iter(
                            self.individual_dataloaders[iterator]
                        )
                        # pyrefly: ignore [unsupported-operation]
                        batch_dict[iterator] = next(self.individual_iterators[iterator])

        if len(batch_dict) == 0:
            raise StopIteration
        return batch_dict


@dataclass
class RandomizedBatchSampler(DataIterationStrategy):
    weights: Optional[Dict[str, float]] = None
    stopping_mechanism: StoppingMechanism = StoppingMechanism.ALL_DATASETS_EXHAUSTED
    enforce_same_loader_across_ranks: bool = False


class RandomizedBatchSamplerIterator(MultiIterator):
    """RandomizedBatchSamplerIterator randomly samples from each dataset
    using the provided weights.

    By default, the iterator stops after all datasets are exhausted. This can be changed
    by setting another stopping mechanism.

    Returns batches of the format: {dataloader_name: batch_from_dataloader}

    Args:
        individual_dataloaders (Mapping[str, Union[DataLoader, Iterable]]): A mapping of DataLoaders or Iterables with dataloader name as key
        and dataloader/iterable object as value.
        iteration_strategy (RandomizedBatchSampler): A RandomizedBatchSampler dataclass indicating how the dataloaders are iterated over.

    Examples:
        >>> loaders = {'a': torch.utils.data.DataLoader(range(4), batch_size=4),
            'b': torch.utils.data.DataLoader(range(15), batch_size=5)}
        >>> randomized_batch_sampler = RandomizedBatchSampler(
                stopping_mechanism=StoppingMechanism.ALL_DATASETS_EXHAUSTED
            )
        >>> combined_iterator = RandomizedBatchSamplerIterator(loaders, randomized_batch_sampler)
        >>> for item in combined_iterator:
                print(item)
        {'b': tensor([0, 1, 2, 3, 4])}
        {'b': tensor([5, 6, 7, 8, 9])}
        {'a': tensor([0, 1, 2, 3])}
        {'b': tensor([10, 11, 12, 13, 14])}

    """

    def __init__(
        self,
        individual_dataloaders: Mapping[str, Union[DataLoader, Iterable[object]]],
        iteration_strategy: RandomizedBatchSampler,
    ) -> None:
        super().__init__(individual_dataloaders, iteration_strategy)
        self._iteration_strategy = iteration_strategy
        self._individual_dataloaders = individual_dataloaders
        self._individual_iterators: MutableMapping[
            str, Union[Iterator[DataLoader], Iterator[object]]
        ] = {name: iter(dl) for name, dl in self._individual_dataloaders.items()}
        self._iterator_names: List[str] = sorted(self._individual_dataloaders.keys())
        weights = iteration_strategy.weights
        if weights is None:
            self._iterator_weights: Optional[List[float]] = None
        else:
            assert set(self._iterator_names).issubset(
                weights.keys()
            ), "Weight keys must match dataloader keys"
            self._iterator_weights = [
                float(weights[name]) for name in self._iterator_names
            ]
        self._iterator_is_exhausted: List[bool] = [False] * len(self._iterator_names)
        self.stopping_mechanism: Optional[StoppingMechanism] = (
            iteration_strategy.stopping_mechanism
        )
        self.enforce_same_loader_across_ranks: bool = (
            iteration_strategy.enforce_same_loader_across_ranks
        )
        if self.enforce_same_loader_across_ranks:
            self._iterator_names_dict: Dict[str, torch.IntTensor] = {
                name: torch.IntTensor([idx])
                for idx, name in enumerate(self._iterator_names)
            }
            self._process_group: dist.ProcessGroup = dist.new_group(
                backend="gloo", ranks=None
            )

        self._iterators_finished: List[str] = []

    def __next__(self) -> Dict[str, Any]:
        if (
            self.stopping_mechanism == StoppingMechanism.SMALLEST_DATASET_EXHAUSTED
            and any(self._iterator_is_exhausted)
        ):
            raise StopIteration
        elif (
            self.stopping_mechanism == StoppingMechanism.ALL_DATASETS_EXHAUSTED
            and all(self._iterator_is_exhausted)
        ):
            raise StopIteration
        elif (
            self.stopping_mechanism
            == StoppingMechanism.RESTART_UNTIL_ALL_DATASETS_EXHAUSTED
            and len(self._iterators_finished) == len(self._individual_iterators)
        ):
            raise StopIteration

        iterator_names = self._iterator_names
        iterator_weights = self._iterator_weights

        if (
            self.stopping_mechanism != StoppingMechanism.WRAP_AROUND_UNTIL_KILLED
            and iterator_weights is not None
        ):
            iterator_names = [
                name
                for name, exhausted in zip(iterator_names, self._iterator_is_exhausted)
                if not exhausted
            ]
            iterator_weights = [
                weight
                for weight, exhausted in zip(
                    iterator_weights, self._iterator_is_exhausted
                )
                if not exhausted
            ]

        selected_key = random.choices(iterator_names, weights=iterator_weights)[0]

        if (
            self.enforce_same_loader_across_ranks
            and dist.is_available()
            and dist.is_initialized()
        ):
            key_idx = self._iterator_names_dict[selected_key]
            dist.broadcast(key_idx, 0, group=self._process_group)
            selected_key = self._iterator_names[cast(int, key_idx.item())]

        try:
            batch = next(self._individual_iterators[selected_key])
        except StopIteration:
            if (
                self.stopping_mechanism
                == StoppingMechanism.RESTART_UNTIL_ALL_DATASETS_EXHAUSTED
            ):
                if selected_key not in self._iterators_finished:
                    self._iterators_finished.append(selected_key)
                if len(self._iterators_finished) == len(self._individual_iterators):
                    raise StopIteration
                else:
                    self._individual_iterators[selected_key] = iter(
                        self._individual_dataloaders[selected_key]
                    )
                batch = next(self._individual_iterators[selected_key])
            elif self.stopping_mechanism == StoppingMechanism.WRAP_AROUND_UNTIL_KILLED:
                self._individual_iterators[selected_key] = iter(
                    self._individual_dataloaders[selected_key]
                )
                batch = next(self._individual_iterators[selected_key])
            else:
                selected_index = self._iterator_names.index(selected_key)
                self._iterator_is_exhausted[selected_index] = True
                return next(self)

        return {selected_key: batch}


@dataclass
class InOrder(DataIterationStrategy):
    iteration_order: Optional[List[str]] = None


class InOrderIterator(MultiIterator):
    """InOrderIterator returns all batches from a single dataset
    till it is exhausted and then moves to the next one.

    By default, the order is same as the keys of the input
    dataloader dict. This can be overridden to provide custom order.
    Repetition is supported.

    Returns batches of the format: {dataloader_name: batch_from_dataloader}

    Args:
        individual_dataloaders (Mapping[str, Union[DataLoader, Iterable]]): A mapping of DataLoaders or Iterables with dataloader name as key
        and dataloader/iterable object as value.
        iteration_strategy (RandomizedBatchSampler): A RandomizedBatchSampler dataclass indicating how the dataloaders are iterated over.

    Examples:
        >>> loaders = {'a': torch.utils.data.DataLoader(range(4), batch_size=4),
            'b': torch.utils.data.DataLoader(range(15), batch_size=5)}
        >>> in_order_strategy = InOrder()
        >>> combined_iterator = RandomizedBatchSamplerIterator(loaders, in_order_strategy)
        >>> for item in combined_iterator:
                print(item)
        {'a': tensor([0, 1, 2, 3])}
        {'b': tensor([0, 1, 2, 3, 4])}
        {'b': tensor([5, 6, 7, 8, 9])}
        {'b': tensor([10, 11, 12, 13, 14])}

    """

    def __init__(
        self,
        individual_dataloaders: Mapping[str, Union[DataLoader, Iterable[object]]],
        iteration_strategy: InOrder,
    ) -> None:
        super().__init__(individual_dataloaders, iteration_strategy)
        self.iteration_order: List[str] = iteration_strategy.iteration_order or list(
            self.individual_dataloaders.keys()
        )
        self.cur_iter: Union[Iterator[DataLoader], Iterator[object]] = iter(
            self.individual_dataloaders[self.iteration_order[0]]
        )
        self.cur_iterator_idx: int = 0
        self.cur_iterator: str = self.iteration_order[0]
        self.num_iterators: int = len(self.iteration_order)
        self.iterators_finished: int = 0

    def __next__(self) -> Dict[str, Any]:
        if self.iterators_finished == self.num_iterators:
            raise StopIteration

        # If the current iterator doesn't match the expected number of finished iterators,
        # it means we restored from checkpoint and we need to initialize expected iterator
        # This is to avoid calling iter() in the load_state_dict() function.
        if self.iterators_finished != self.cur_iterator_idx:
            logger.info(
                f"Initializing iterator {self.cur_iterator} after resuming from checkpoint"
            )
            self.cur_iterator_idx = self.iterators_finished
            self.cur_iterator = self.iteration_order[self.iterators_finished]
            self.cur_iter = iter(self.individual_dataloaders[self.cur_iterator])

        try:
            return {self.cur_iterator: next(self.cur_iter)}
        except StopIteration:
            self.iterators_finished += 1

            # Raise exception when all iterators are finished
            if self.iterators_finished == self.num_iterators:
                raise StopIteration

            self.cur_iterator_idx += 1
            self.cur_iterator = self.iteration_order[self.iterators_finished]
            self.cur_iter = iter(self.individual_dataloaders[self.cur_iterator])

            return self.__next__()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "iterators_finished": self.iterators_finished,
            "cur_iterator": self.cur_iterator,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        iterators_finished: int = state_dict["iterators_finished"]
        cur_iterator: str = state_dict["cur_iterator"]
        logger.info(
            f"Loading InOrderIterator state. Trying to set iterators_finished to {iterators_finished} to restore {cur_iterator}"
        )

        if cur_iterator not in self.iteration_order or iterators_finished > len(
            self.iteration_order
        ):
            logger.warning(
                f"Will not restore InOrderIterator state, since expected dataloader was not found in available iterators: {cur_iterator}"
            )
            return

        self.iterators_finished = iterators_finished
        # We do not initialize actual iterator here to avoid checkpoint restore taking longer


class DataIterationStrategyRegistry:
    """A generic iterator registry.

    This will be used to provide default iterators.
    """

    REGISTRY = {
        RoundRobin: RoundRobinIterator,
        AllDatasetBatches: AllDatasetBatchesIterator,
        RandomizedBatchSampler: RandomizedBatchSamplerIterator,
        InOrder: InOrderIterator,
    }

    @classmethod
    def get(cls, iteration_strategy: DataIterationStrategy) -> Type[MultiIterator]:
        if iteration_strategy.__class__ in cls.REGISTRY:
            return cls.REGISTRY[iteration_strategy.__class__]
        raise NotImplementedError(
            f"No iterator implementation for {iteration_strategy}"
        )
