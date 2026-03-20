# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Union

from torchtnt.framework.state import State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTestUnit, TTrainUnit


class Callback:
    """
    A Callback is an optional extension that can be used to supplement your loop with additional functionality. Good candidates
    for such logic are ones that can be re-used across units. Callbacks are generally not intended for modeling code; this should go
    in your `Unit <https://www.internalfb.com/intern/staticdocs/torchtnt/framework/unit.html>`_. To write your own callback,
    subclass the Callback class and add your own code into the hooks.

    Below is an example of a basic callback which prints a message at various points during execution.

    .. code-block:: python

      from torchtnt.framework.callback import Callback
      from torchtnt.framework.state import State
      from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit

      class PrintingCallback(Callback):
          def on_train_start(self, state: State, unit: TTrainUnit) -> None:
              print("Starting training")

          def on_train_end(self, state: State, unit: TTrainUnit) -> None:
              print("Ending training")

          def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
              print("Starting evaluation")

          def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
              print("Ending evaluation")

          def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
              print("Starting prediction")

          def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
              print("Ending prediction")

    To use a callback, instantiate the class and pass it in the ``callbacks`` parameter to the :py:func:`~torchtnt.framework.train`, :py:func:`~torchtnt.framework.evaluate`,
    :py:func:`~torchtnt.framework.predict`, or :py:func:`~torchtnt.framework.fit` entry point.

    .. code-block:: python

      printing_callback = PrintingCallback()
      train(train_unit, train_dataloader, callbacks=[printing_callback])
    """

    @property
    def name(self) -> str:
        """A distinct name per instance. This is useful for debugging, profiling, and checkpointing purposes."""
        return self.__class__.__qualname__

    def on_exception(
        self,
        state: State,
        unit: Union[TTrainUnit, TEvalUnit, TPredictUnit, TTestUnit],
        exc: BaseException,
    ) -> None:
        """Hook called when an exception occurs."""
        pass

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        """Hook called before training starts."""
        pass

    def on_train_epoch_start(self, state: State, unit: TTrainUnit) -> None:
        """Hook called before a new train epoch starts."""
        pass

    def on_train_dataloader_iter_creation_start(
        self, state: State, unit: TTrainUnit
    ) -> None:
        """Hook called before the dataloader iterator is created."""
        pass

    def on_train_dataloader_iter_creation_end(
        self, state: State, unit: TTrainUnit
    ) -> None:
        """Hook called after the dataloader iterator is created."""
        pass

    def on_train_get_next_batch_start(self, state: State, unit: TTrainUnit) -> None:
        """Hook called before getting the data batch for the next train step."""
        pass

    def on_train_get_next_batch_end(self, state: State, unit: TTrainUnit) -> None:
        """Hook called after getting the data batch for the next train step."""
        pass

    def on_train_step_start(self, state: State, unit: TTrainUnit) -> None:
        """Hook called before a new train step starts."""
        pass

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        """Hook called after a train step ends."""
        pass

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        """Hook called after a train epoch ends."""
        pass

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        """Hook called after training ends."""
        pass

    def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
        """Hook called before evaluation starts."""
        pass

    def on_eval_epoch_start(self, state: State, unit: TEvalUnit) -> None:
        """Hook called before a new eval epoch starts."""
        pass

    def on_eval_dataloader_iter_creation_start(
        self, state: State, unit: TEvalUnit
    ) -> None:
        """Hook called before the dataloader iterator is created."""
        pass

    def on_eval_dataloader_iter_creation_end(
        self, state: State, unit: TEvalUnit
    ) -> None:
        """Hook called after the dataloader iterator is created."""
        pass

    def on_eval_get_next_batch_start(self, state: State, unit: TEvalUnit) -> None:
        """Hook called before getting the data batch for the next eval step."""
        pass

    def on_eval_get_next_batch_end(self, state: State, unit: TEvalUnit) -> None:
        """Hook called after getting the data batch for the next eval step."""
        pass

    def on_eval_step_start(self, state: State, unit: TEvalUnit) -> None:
        """Hook called before a new eval step starts."""
        pass

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        """Hook called after an eval step ends."""
        pass

    def on_eval_epoch_end(self, state: State, unit: TEvalUnit) -> None:
        """Hook called after an eval epoch ends."""
        pass

    def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
        """Hook called after evaluation ends."""
        pass

    def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
        """Hook called before prediction starts."""
        pass

    def on_predict_epoch_start(self, state: State, unit: TPredictUnit) -> None:
        """Hook called before a new predict epoch starts."""
        pass

    def on_predict_dataloader_iter_creation_start(
        self, state: State, unit: TPredictUnit
    ) -> None:
        """Hook called before the dataloader iterator is created."""
        pass

    def on_predict_dataloader_iter_creation_end(
        self, state: State, unit: TPredictUnit
    ) -> None:
        """Hook called after the dataloader iterator is created."""
        pass

    def on_predict_get_next_batch_start(self, state: State, unit: TPredictUnit) -> None:
        """Hook called before getting the data batch for the next predict step."""
        pass

    def on_predict_get_next_batch_end(self, state: State, unit: TPredictUnit) -> None:
        """Hook called after getting the data batch for the next predict step."""
        pass

    def on_predict_step_start(self, state: State, unit: TPredictUnit) -> None:
        """Hook called before a new predict step starts."""
        pass

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        """Hook called after a predict step ends."""
        pass

    def on_predict_epoch_end(self, state: State, unit: TPredictUnit) -> None:
        """Hook called after a predict epoch ends."""
        pass

    def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
        """Hook called after prediction ends."""
        pass

    def on_test_start(self, state: State, unit: TTestUnit) -> None:
        """Hook called before testing starts."""
        pass

    def on_test_epoch_start(self, state: State, unit: TTestUnit) -> None:
        """Hook called before a new test epoch starts."""
        pass

    def on_test_dataloader_iter_creation_start(
        self, state: State, unit: TTestUnit
    ) -> None:
        """Hook called before the dataloader iterator is created."""
        pass

    def on_test_dataloader_iter_creation_end(
        self, state: State, unit: TTestUnit
    ) -> None:
        """Hook called after the dataloader iterator is created."""
        pass

    def on_test_get_next_batch_start(self, state: State, unit: TTestUnit) -> None:
        """Hook called before getting the data batch for the next test step."""
        pass

    def on_test_get_next_batch_end(self, state: State, unit: TTestUnit) -> None:
        """Hook called after getting the data batch for the next test step."""
        pass

    def on_test_step_start(self, state: State, unit: TTestUnit) -> None:
        """Hook called before a new test step starts."""
        pass

    def on_test_step_end(self, state: State, unit: TTestUnit) -> None:
        """Hook called after a test step ends."""
        pass

    def on_test_epoch_end(self, state: State, unit: TTestUnit) -> None:
        """Hook called after a test epoch ends."""
        pass

    def on_test_end(self, state: State, unit: TTestUnit) -> None:
        """Hook called after testing ends."""
        pass
