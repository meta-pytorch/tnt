#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Iterator, Optional, Type, TYPE_CHECKING, Union

from pyre_extensions import none_throws
from torchtnt.utils.data.iterators import (
    DataIterationStrategy,
    DataIterationStrategyRegistry,
    MultiIterator,
)
from torchtnt.utils.stateful import Stateful


if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger: logging.Logger = logging.getLogger(__name__)


class MultiDataLoader:
    """MultiDataLoader cycles through individual dataloaders passed to it.

    Attributes:
        individual_dataloaders (Dict[str, Union[DataLoader, Iterable]]): A dictionary of DataLoaders or Iterables with dataloader name as key
        and dataloader/iterable object as value.
        iteration_strategy (DataIterationStrategy): A dataclass indicating how the dataloaders are iterated over.
        iterator_cls (MultiIterator, optional): A subclass of MultiIterator defining iteration logic. This is the type, not an object instance
        ignore_empty_data (bool): skip dataloaders which contain no data. It's False by default, and an exception is raised.

    Note:
        `TorchData <https://pytorch.org/data/beta/index.html>`_ also has generic
        multi-data sources reading support to achieve the same functionality
        provided by MultiIterator.
        For example, `mux`, `mux_longest`, `cycle`, `zip` etc. Please refer
        to the documentation for more details.
    """

    def __init__(
        self,
        individual_dataloaders: Dict[str, Union[DataLoader, Iterable[object]]],
        iteration_strategy: DataIterationStrategy,
        iterator_cls: Optional[Type[MultiIterator]] = None,
        ignore_empty_data: bool = False,
    ) -> None:
        self.individual_dataloaders = individual_dataloaders
        self.iteration_strategy = iteration_strategy
        self.iterator_cls = iterator_cls
        self.current_iterator: Optional[MultiIterator] = None
        for name in list(individual_dataloaders.keys()):
            try:
                next(iter(self.individual_dataloaders[name]))
            except StopIteration:
                if not ignore_empty_data:
                    raise ValueError(f"Dataloader '{name}' contains no data.")
                else:
                    logger.warning(
                        f"Dataloader '{name}' which contains no data. "
                        "You might have empty dataloaders in the input dict."
                    )
        self.iterator_state: Optional[Dict[str, Any]] = None

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterator functions for the collection of dataloaders.

        Returns:
            a newly created iterator based on DataIterationStrategy

        """
        iterator_cls = self.iterator_cls
        if iterator_cls is None:
            iterator_cls = DataIterationStrategyRegistry.get(self.iteration_strategy)
        # in practice, DataIterationStrategyRegistry.get() returns just concrete classes
        # pyre-ignore[45]: Cannot instantiate abstract class `MultiIterator`.
        self.current_iterator = iterator_cls(
            individual_dataloaders=self.individual_dataloaders,
            iteration_strategy=self.iteration_strategy,
        )
        if self.iterator_state is not None:
            self.current_iterator.load_state_dict(self.iterator_state)

        self.iterator_state = None
        return none_throws(self.current_iterator)

    def state_dict(self) -> Dict[str, Any]:
        """Return an aggregated state dict based on individual dataloaders.

        The state dict is keyed off the names provided by ``individual_dataloaders``.

        Note:
            Only states from dataloaders that implement the :class:`~torchtnt.utils.stateful.Stateful` protocol are included in the returned state dict.
        """
        state_dict = {}
        for name, dl in self.individual_dataloaders.items():
            if isinstance(dl, Stateful):
                state_dict[name] = dl.state_dict()

        if (current_iterator := self.current_iterator) is not None:
            iterator_state = current_iterator.state_dict()
            if iterator_state:
                logger.info("Storing iterator state in MultiDataLoader state_dict")
                # we make an implicit assumption here that none of the dataloaders have the "iterator_state" key in order to be backwards compatible
                # with already saved checkpoints (we don't want to modify the dataloaders stateful names)
                state_dict["iterator_state"] = iterator_state

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads aggregated state dict based on individual dataloaders.

        The provided state dict should be keyed off the names provided by ``individual_dataloaders``.

        Note:
            Only states from dataloaders that implement the :class:`~torchtnt.utils.stateful.Stateful` protocol are loaded.
        """
        for name, dl in self.individual_dataloaders.items():
            if isinstance(dl, Stateful):
                contents = state_dict.get(name, None)
                if contents is None:
                    logger.warning(
                        f"Skipping loading state dict for dataloader {name} as there is no corresponding entry in the state dict"
                    )
                    continue
                dl.load_state_dict(contents)

        if "iterator_state" in state_dict:
            # this will be used during the next __iter__ call
            self.iterator_state = state_dict["iterator_state"]
