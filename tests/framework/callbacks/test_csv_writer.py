#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import tempfile
import unittest
from typing import Any, List, Union

from torchtnt.framework._test_utils import DummyPredictUnit, generate_random_dataloader
from torchtnt.framework.callbacks.base_csv_writer import BaseCSVWriter
from torchtnt.framework.predict import predict
from torchtnt.framework.state import State
from torchtnt.framework.unit import PredictUnit, TPredictData

_HEADER_ROW = ["output"]
_FILENAME = "test_csv_writer.csv"


class CustomCSVWriter(BaseCSVWriter):
    def get_step_output_rows(
        self,
        state: State,
        unit: PredictUnit[TPredictData],
        step_output: Any,
    ) -> Union[List[str], List[List[str]]]:
        return [["1"], ["2"]]


class CustomCSVWriterSingleRow(BaseCSVWriter):
    def get_step_output_rows(
        self,
        state: State,
        unit: PredictUnit[TPredictData],
        step_output: Any,
    ) -> Union[List[str], List[List[str]]]:
        return ["1"]


class BaseCSVWriterTest(unittest.TestCase):
    def test_csv_writer(self) -> None:
        """
        Test BaseCSVWriter callback that creates file from multiple lists with predict
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2

        my_unit = DummyPredictUnit(2)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_callback = CustomCSVWriter(
                header_row=_HEADER_ROW, dir_path=temp_dir, filename=_FILENAME
            )

            predict(my_unit, dataloader, callbacks=[csv_callback])

            # Check file exists and is successfully opened
            csv_file = f"{temp_dir}/{_FILENAME}"
            self.assertEqual(csv_callback.output_path, csv_file)
            self.assertIsNotNone(csv_callback._file)

    def test_csv_writer_single_row(self) -> None:
        """
        Test BaseCSVWriter callback that creates file from single list with predict
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2

        my_unit = DummyPredictUnit(2)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_callback = CustomCSVWriterSingleRow(
                header_row=_HEADER_ROW, dir_path=temp_dir, filename=_FILENAME
            )
            predict(my_unit, dataloader, callbacks=[csv_callback])

            # Check file exists and is successfully opened
            csv_file = f"{temp_dir}/{_FILENAME}"
            self.assertEqual(csv_callback.output_path, csv_file)
            self.assertIsNotNone(csv_callback._file)

    def test_csv_writer_with_no_output_rows_def(self) -> None:
        """
        Test BaseCSVWriter callback without output defined
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2

        my_unit = DummyPredictUnit(2)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        # Throw exception because get_step_output_rows is not defined.
        with self.assertRaises(TypeError):
            # pyre-fixme[45]: Cannot instantiate abstract class `BaseCSVWriter`.
            csv_callback = BaseCSVWriter(
                header_row=_HEADER_ROW, dir_path="", filename=_FILENAME
            )
            predict(my_unit, dataloader, callbacks=[csv_callback])
