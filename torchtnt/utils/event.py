# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from typing import Dict, Union

EventMetadataValue = Union[str, int, float, bool, None]


@dataclass
class Event:
    """
    The class represents the generic event that occurs during a TorchTNT
    loop. The event can be any kind of meaningful action.

    Args:
        name: event name.
        metadata: additional data that is associated with the event.
    """

    name: str
    metadata: Dict[str, EventMetadataValue] = field(default_factory=dict)
