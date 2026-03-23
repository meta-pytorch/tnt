# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import csv
import os
from abc import ABC, abstractmethod
from typing import Any, List, TextIO, Union

from pyre_extensions import none_throws
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import EntryPoint, State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTestUnit, TTrainUnit
from torchtnt.utils import get_filesystem, get_global_rank

DEFAULT_FILE_NAME = "predictions.csv"


class BaseCSVWriter(Callback, ABC):
    """
    A callback to write prediction outputs to a CSV file.

    This callback provides an interface to simplify writing outputs during prediction
    into a CSV file. This callback must be extended with an implementation for
    ``get_step_output_rows`` to write the desired outputs as rows in the CSV file.

    By default, outputs at each step across all processes will be written into the same CSV file.
    The outputs in each row is a a list of strings, and should match
    the columns names defined in ``header_row``.

    Args:
        header_row: columns of the CSV file
        dir_path: directory path of where to save the CSV file
        delimiter: separate columns in one row. Default is tab
        filename: name of the file. Default filename is "predictions.csv"
    """

    def __init__(
        self,
        header_row: List[str],
        dir_path: str,
        delimiter: str = "\t",
        filename: str = DEFAULT_FILE_NAME,
    ) -> None:
        super().__init__()
        self.header_row = header_row
        self.delimiter = delimiter

        self.output_path: str = os.path.join(dir_path, filename)
        fs = get_filesystem(self.output_path)
        self._file: TextIO = fs.open(self.output_path, mode="a")
        self._writer: csv._writer = csv.writer(self._file, delimiter=delimiter)

    @abstractmethod
    def get_step_output_rows(
        self,
        state: State,
        unit: TPredictUnit,
        step_output: Any,
    ) -> Union[List[str], List[List[str]]]: ...

    def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
        if get_global_rank() == 0:
            self._writer.writerow(self.header_row)

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        predict_state = none_throws(state.predict_state)
        step_output = predict_state.step_output
        output_rows = self.get_step_output_rows(state, unit, step_output)

        # Check whether the first item is a list or not
        if len(output_rows) > 0:
            if isinstance(output_rows[0], list):
                for row in output_rows:
                    self._writer.writerow(row)
            else:
                self._writer.writerow(output_rows)

    def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
        self._file.flush()
        self._file.close()

    def on_exception(
        self,
        state: State,
        unit: Union[TTrainUnit, TEvalUnit, TPredictUnit, TTestUnit],
        exc: BaseException,
    ) -> None:
        if state.entry_point == EntryPoint.PREDICT:
            self._file.flush()
            self._file.close()
