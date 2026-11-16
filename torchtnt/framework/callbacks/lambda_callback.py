# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, Optional, Union

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit


class Lambda(Callback):
    """
    A callback that accepts functions run during the training, evaluation, and prediction loops.

    Args:

        on_exception: function to run when an exception occurs.
        on_train_start: function to run when train starts.
        on_train_epoch_start: function to run when each train epoch starts.
        on_train_get_next_batch_end: function to run when the data batch for the next train step is retrieved.
        on_train_step_start: function to run when each train step starts.
        on_train_step_end: function to run when each train step ends.
        on_train_epoch_end: function to run when each train epoch ends.
        on_train_end: function to run when train ends.
        on_eval_start: function to run when eval starts.
        on_eval_epoch_start: function to run when each eval epoch starts.
        on_eval_get_next_batch_end: function to run when the data batch for the next eval step is retrieved.
        on_eval_step_start: function to run when each eval step starts.
        on_eval_step_end: function to run when each eval step ends.
        on_eval_epoch_end: function to run when each eval epoch ends.
        on_eval_end: function to run when eval ends.
        on_predict_start: function to run when predict starts.
        on_predict_epoch_start: function to run when each predict epoch starts.
        on_predict_get_next_batch_end: function to run when the data batch for the next predict step is retrieved.
        on_predict_step_start: function to run when each predict step starts.
        on_predict_step_end: function to run when each predict step ends.
        on_predict_epoch_end: function to run when each predict epoch ends.
        on_predict_end: function to run when predict ends.

    Examples::

        from torchtnt.framework.callbacks import Lambda
        from torchtnt.framework.evaluate import evaluate

        dataloader = MyDataLoader()
        unit = MyUnit()

        def print_on_step_start(state, unit) -> None:
            print(f'starting eval step {unit.eval_progress.num_steps_completed}')


        lambda_cb = Lambda(
            on_eval_start=lambda *args, print('starting eval'),
            on_eval_step_start=print_on_step_start,
        )

        evaluate(unit, dataloader, callbacks=[lambda_cb])

    """

    def __init__(
        self,
        *,
        on_exception: Optional[
            Callable[
                [
                    State,
                    Union[
                        TTrainUnit,
                        TEvalUnit,
                        TPredictUnit,
                    ],
                    BaseException,
                ],
                None,
            ]
        ] = None,
        on_train_start: Optional[Callable[[State, TTrainUnit], None]] = None,
        on_train_epoch_start: Optional[Callable[[State, TTrainUnit], None]] = None,
        on_train_dataloader_iter_creation_start: Optional[
            Callable[[State, TTrainUnit], None]
        ] = None,
        on_train_dataloader_iter_creation_end: Optional[
            Callable[[State, TTrainUnit], None]
        ] = None,
        on_train_get_next_batch_start: Optional[
            Callable[[State, TTrainUnit], None]
        ] = None,
        on_train_get_next_batch_end: Optional[
            Callable[[State, TTrainUnit], None]
        ] = None,
        on_train_step_start: Optional[Callable[[State, TTrainUnit], None]] = None,
        on_train_step_end: Optional[Callable[[State, TTrainUnit], None]] = None,
        on_train_epoch_end: Optional[Callable[[State, TTrainUnit], None]] = None,
        on_train_end: Optional[Callable[[State, TTrainUnit], None]] = None,
        on_eval_start: Optional[Callable[[State, TEvalUnit], None]] = None,
        on_eval_epoch_start: Optional[Callable[[State, TEvalUnit], None]] = None,
        on_eval_dataloader_iter_creation_start: Optional[
            Callable[[State, TTrainUnit], None]
        ] = None,
        on_eval_dataloader_iter_creation_end: Optional[
            Callable[[State, TTrainUnit], None]
        ] = None,
        on_eval_get_next_batch_start: Optional[
            Callable[[State, TTrainUnit], None]
        ] = None,
        on_eval_get_next_batch_end: Optional[Callable[[State, TEvalUnit], None]] = None,
        on_eval_step_start: Optional[Callable[[State, TEvalUnit], None]] = None,
        on_eval_step_end: Optional[Callable[[State, TEvalUnit], None]] = None,
        on_eval_epoch_end: Optional[Callable[[State, TEvalUnit], None]] = None,
        on_eval_end: Optional[Callable[[State, TEvalUnit], None]] = None,
        on_predict_start: Optional[Callable[[State, TPredictUnit], None]] = None,
        on_predict_epoch_start: Optional[Callable[[State, TPredictUnit], None]] = None,
        on_predict_dataloader_iter_creation_start: Optional[
            Callable[[State, TTrainUnit], None]
        ] = None,
        on_predict_dataloader_iter_creation_end: Optional[
            Callable[[State, TTrainUnit], None]
        ] = None,
        on_predict_get_next_batch_start: Optional[
            Callable[[State, TTrainUnit], None]
        ] = None,
        on_predict_get_next_batch_end: Optional[
            Callable[[State, TPredictUnit], None]
        ] = None,
        on_predict_step_start: Optional[Callable[[State, TPredictUnit], None]] = None,
        on_predict_step_end: Optional[Callable[[State, TPredictUnit], None]] = None,
        on_predict_epoch_end: Optional[Callable[[State, TPredictUnit], None]] = None,
        on_predict_end: Optional[Callable[[State, TPredictUnit], None]] = None,
    ) -> None:
        for k, v in locals().items():
            if k == "self":
                continue
            if v is not None:
                setattr(self, k, v)
