#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest import mock
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from pyre_extensions import none_throws
from torch import nn
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE, StorageMeta
from torch.utils.data import DataLoader
from torchsnapshot.test_utils import assert_state_dict_eq, check_state_dict_eq
from torchtnt.framework._test_utils import (
    DummyAutoUnit,
    DummyEvalUnit,
    DummyMeanMetric,
    DummyMultiOptimUnit,
    DummyPredictUnit,
    DummyTrainUnit,
    generate_dummy_stateful_dataloader,
    generate_random_dataloader,
    get_dummy_train_state,
)
from torchtnt.framework.callbacks.checkpointer_types import RestoreOptions
from torchtnt.framework.callbacks.dcp_saver import DistributedCheckpointSaver
from torchtnt.framework.evaluate import evaluate
from torchtnt.framework.fit import fit
from torchtnt.framework.predict import predict
from torchtnt.framework.state import State
from torchtnt.framework.train import train
from torchtnt.utils.checkpoint import BestCheckpointConfig, get_latest_checkpoint_path
from torchtnt.utils.distributed import get_global_rank, spawn_multi_process
from torchtnt.utils.env import seed
from torchtnt.utils.test_utils import skip_if_not_distributed


class DistributedCheckpointSaverTest(unittest.TestCase):
    def tearDown(self) -> None:
        # needed for github test, to reset WORLD_SIZE env var
        os.environ.pop("WORLD_SIZE", None)

    def test_save_restore(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2
        save_every_n_train_steps = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        expected_paths: List[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            expected_paths = [
                f"{temp_dir}/epoch_0_train_step_2",
                f"{temp_dir}/epoch_0_train_step_4",
                f"{temp_dir}/epoch_1_train_step_6",
                f"{temp_dir}/epoch_1_train_step_8",
                f"{temp_dir}/epoch_1_train_step_10",
                f"{temp_dir}/epoch_2_train_step_10",  # extra checkpoint on_train_end
            ]
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
            )

            saved_checkpoint_paths: List[str] = []

            def _checkpoint_save_callback(state: State, checkpoint_id: str) -> None:
                saved_checkpoint_paths.append(checkpoint_id)

            my_unit.on_checkpoint_save = _checkpoint_save_callback  # pyre-ignore

            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[dcp_cb])

            end_num_steps_completed = my_unit.train_progress.num_steps_completed
            self.assertEqual(saved_checkpoint_paths, expected_paths)
            dcp_cb.restore(expected_paths[0], my_unit)
            restored_num_steps_completed = my_unit.train_progress.num_steps_completed
            # A snapshot is saved every n steps
            # so the first snapshot's progress will be equal to save_every_n_train_steps
            self.assertNotEqual(restored_num_steps_completed, end_num_steps_completed)
            self.assertEqual(restored_num_steps_completed, save_every_n_train_steps)

    def test_save_restore_dataloader_state(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        save_every_n_train_steps = 2
        max_steps = 3

        my_unit = DummyTrainUnit(input_dim=input_dim)
        stateful_dataloader = DummyStatefulDataLoader(
            dataloader=generate_random_dataloader(dataset_len, input_dim, batch_size)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
            )
            train(
                my_unit,
                stateful_dataloader,
                max_steps=max_steps,
                callbacks=[dcp_cb],
            )
            # state_dict has been called once on dataloader
            self.assertEqual(stateful_dataloader.state_dict_call_count, 1)
            self.assertEqual(stateful_dataloader.load_state_dict_call_count, 0)

            # restoring from first checkpoint, has dataloader in manifest
            dcp_cb.restore(
                temp_dir + f"/epoch_{0}_train_step_{save_every_n_train_steps}",
                my_unit,
                train_dataloader=stateful_dataloader,
            )
            # load_state_dict has been called once on dataloader
            self.assertEqual(stateful_dataloader.load_state_dict_call_count, 1)

            # restoring from last checkpoint (on train end), does not have dataloader state in manifest

            with self.assertLogs(level="WARNING") as log:
                dcp_cb.restore(
                    temp_dir + f"/epoch_{1}_train_step_{max_steps}",
                    my_unit,
                    train_dataloader=stateful_dataloader,
                )
                # load_state_dict is not called again on dataloader because there is no dataloader in manifest
                self.assertEqual(stateful_dataloader.load_state_dict_call_count, 1)
                self.assertEqual(
                    log.output[0],
                    "WARNING:torchtnt.framework.callbacks.dcp_saver:dataloader (train) was passed to `restore` but no dataloader exists in checkpoint metadata.",
                )

    def test_restore_from_latest(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 1
        save_every_n_train_steps = 2
        expected_steps_per_epoch = math.ceil(dataset_len / batch_size)

        my_unit = DummyTrainUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        with tempfile.TemporaryDirectory() as temp_dir:
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[dcp_cb])

            with mock.patch(
                "torchtnt.framework.callbacks.dcp_saver.DistributedCheckpointSaver.restore"
            ) as mock_restore:
                restored = dcp_cb.restore_from_latest(temp_dir, my_unit, no_dist=True)
                self.assertIn(
                    temp_dir
                    + f"/epoch_{max_epochs}_train_step_{expected_steps_per_epoch}",
                    mock_restore.call_args.args,
                )
                self.assertTrue(restored)

    def test_save_restore_no_train_progress(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2
        expected_steps_per_epoch = math.ceil(dataset_len / batch_size)
        save_every_n_train_steps = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        expected_paths: List[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            cumulative_steps = 0
            for epoch in range(max_epochs):
                for _ in range(
                    save_every_n_train_steps,
                    expected_steps_per_epoch + 1,
                    save_every_n_train_steps,
                ):
                    cumulative_steps += save_every_n_train_steps
                    expected_paths.append(
                        os.path.join(
                            temp_dir, f"epoch_{epoch}_train_step_{cumulative_steps}"
                        )
                    )
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[dcp_cb])

            end_num_steps_completed = my_unit.train_progress.num_steps_completed
            self.assertGreater(len(expected_paths), 0)
            dcp_cb.restore(
                expected_paths[0],
                my_unit,
                restore_options=RestoreOptions(restore_train_progress=False),
            )
            restored_num_steps_completed = my_unit.train_progress.num_steps_completed
            # no train progress was restored so the progress after restoration should be the same as the progress before restoration
            self.assertEqual(restored_num_steps_completed, end_num_steps_completed)

    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_save_restore_no_optimizer_restore(self, mock_dist_cp: MagicMock) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        restore_options = RestoreOptions(restore_optimizers=False)
        DistributedCheckpointSaver.restore(
            path="path/to/snapshot",
            unit=my_unit,
            restore_options=restore_options,
        )
        app_state = mock_dist_cp.load.call_args.args[0]["app_state"].state_dict()
        self.assertNotIn("optimizer", app_state)
        DistributedCheckpointSaver.restore(path="path/to/snapshot", unit=my_unit)
        app_state = mock_dist_cp.load.call_args.args[0]["app_state"].state_dict()
        self.assertIn("optimizer", app_state)

    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_save_restore_no_lr_scheduler_restore(
        self, mock_dist_cp: MagicMock
    ) -> None:
        my_unit = DummyAutoUnit(module=nn.Linear(2, 3))
        restore_options = RestoreOptions(restore_lr_schedulers=False)
        DistributedCheckpointSaver.restore(
            path="path/to/snapshot", unit=my_unit, restore_options=restore_options
        )
        app_state = mock_dist_cp.load.call_args.args[0]["app_state"].state_dict()
        self.assertNotIn("lr_scheduler", app_state)
        DistributedCheckpointSaver.restore(path="path/to/snapshot", unit=my_unit)
        app_state = mock_dist_cp.load.call_args.args[0]["app_state"].state_dict()
        self.assertIn("lr_scheduler", app_state)

    @patch("torchtnt.framework.callbacks.dcp_saver._init_optim_state")
    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_save_restore_no_init_optim_state(
        self, _: MagicMock, mock_init_optim_state: MagicMock
    ) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        restore_options = RestoreOptions(init_optim_states=False)
        DistributedCheckpointSaver.restore(
            path="path/to/snapshot",
            unit=my_unit,
            restore_options=restore_options,
        )
        mock_init_optim_state.assert_not_called()

        DistributedCheckpointSaver.restore(
            path="path/to/snapshot",
            unit=my_unit,
            restore_options=RestoreOptions(),
        )
        mock_init_optim_state.assert_called()

    @skip_if_not_distributed
    def test_save_restore_ddp(self) -> None:
        spawn_multi_process(
            2,
            "cpu:gloo,cuda:gloo",
            self._save_restore_ddp,
        )

    @staticmethod
    def _save_restore_ddp() -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2
        save_every_n_epochs = 1
        seed(0)

        my_unit = DummyAutoUnit(module=torch.nn.Linear(input_dim, 2), strategy="ddp")
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        if get_global_rank() == 0:
            temp_dir = tempfile.mkdtemp()
        else:
            temp_dir = ""

        dcp_cb = DistributedCheckpointSaver(
            temp_dir,
            save_every_n_epochs=save_every_n_epochs,
        )
        temp_dir = dcp_cb.dirpath
        train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[dcp_cb])
        tc = unittest.TestCase()
        try:
            my_new_unit = DummyAutoUnit(
                module=torch.nn.Linear(input_dim, 2), strategy="ddp"
            )
            optim_equal = check_state_dict_eq(
                my_new_unit.optimizer.state_dict(), my_unit.optimizer.state_dict()
            )
            tc.assertFalse(optim_equal)
            module_equal = check_state_dict_eq(
                my_new_unit.module.state_dict(), my_unit.module.state_dict()
            )
            tc.assertFalse(module_equal)
            # get latest checkpoint
            ckpt_path = os.path.join(temp_dir, f"epoch_{max_epochs}_train_step_10")
            dcp_cb.restore(ckpt_path, my_new_unit)

            assert_state_dict_eq(
                tc, my_new_unit.optimizer.state_dict(), my_unit.optimizer.state_dict()
            )
            assert_state_dict_eq(
                tc, my_new_unit.module.state_dict(), my_unit.module.state_dict()
            )
        finally:
            dist.barrier()  # avoid race condition
            if get_global_rank() == 0:
                shutil.rmtree(temp_dir)  # delete temp directory

    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_save_default_planner_storage_components(
        self, mock_dist_cp: MagicMock
    ) -> None:
        from torch.distributed.checkpoint._fsspec_filesystem import FsspecWriter

        input_dim = 2
        save_every_n_train_steps = 1

        my_unit = DummyTrainUnit(input_dim=input_dim)
        state = get_dummy_train_state()

        with tempfile.TemporaryDirectory() as temp_dir:
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
                knob_options=KnobOptions(1),
            )

            dcp_cb._checkpoint_impl(
                state=state,
                unit=my_unit,
                checkpoint_id=temp_dir,
                hook="on_train_epoch_end",
            )

            planner = mock_dist_cp.save.call_args_list[0][1]["planner"]
            storage_writer = mock_dist_cp.save.call_args_list[0][1]["storage_writer"]

            self.assertIsInstance(planner, DefaultSavePlanner)
            self.assertIsInstance(storage_writer, FsspecWriter)

    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_save_planner_storage_components(self, mock_dist_cp: MagicMock) -> None:
        input_dim = 2
        save_every_n_train_steps = 1

        my_unit = DummyTrainUnit(input_dim=input_dim)
        state = get_dummy_train_state()

        with tempfile.TemporaryDirectory() as temp_dir:
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
                knob_options=KnobOptions(1),
            )

            dcp_cb._checkpoint_impl(
                state=state,
                unit=my_unit,
                checkpoint_id=temp_dir,
                hook="on_train_epoch_end",
                planner=DummySavePlanner(),
                storage_writer=DummyStorageWriter(path=temp_dir),
            )

            planner = mock_dist_cp.save.call_args_list[0][1]["planner"]
            storage_writer = mock_dist_cp.save.call_args_list[0][1]["storage_writer"]

            self.assertIsInstance(planner, DummySavePlanner)
            self.assertIsInstance(storage_writer, DummyStorageWriter)

    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_restore_default_planner_storage_components(
        self, mock_dist_cp: MagicMock
    ) -> None:
        from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader

        my_unit = DummyTrainUnit(input_dim=2)
        restore_options = RestoreOptions(restore_optimizers=False)
        DistributedCheckpointSaver.restore(
            path="path/to/snapshot",
            unit=my_unit,
            restore_options=restore_options,
        )
        planner = mock_dist_cp.load.call_args[1]["planner"]
        storage_reader = mock_dist_cp.load.call_args[1]["storage_reader"]

        self.assertIsInstance(planner, DefaultLoadPlanner)
        self.assertIsInstance(storage_reader, FsspecReader)

    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_restore_with_id_default_planner_storage_components(
        self, mock_dist_cp: MagicMock
    ) -> None:
        from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader

        my_unit = DummyTrainUnit(input_dim=2)
        restore_options = RestoreOptions(restore_optimizers=False)
        DistributedCheckpointSaver.restore_with_id(
            checkpoint_id="path/to/snapshot",
            unit=my_unit,
            restore_options=restore_options,
        )
        planner = mock_dist_cp.load.call_args[1]["planner"]
        storage_reader = mock_dist_cp.load.call_args[1]["storage_reader"]

        self.assertIsInstance(planner, DefaultLoadPlanner)
        self.assertIsInstance(storage_reader, FsspecReader)

    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_restore_planner_storage_components(self, mock_dist_cp: MagicMock) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        restore_options = RestoreOptions(restore_optimizers=False)
        DistributedCheckpointSaver.restore(
            path="path/to/snapshot",
            unit=my_unit,
            restore_options=restore_options,
            planner=DummyLoadPlanner(),
            storage_reader=DummyStorageReader(path="path/to/snapshot"),
        )
        planner = mock_dist_cp.load.call_args[1]["planner"]
        storage_reader = mock_dist_cp.load.call_args[1]["storage_reader"]

        self.assertIsInstance(planner, DummyLoadPlanner)
        self.assertIsInstance(storage_reader, DummyStorageReader)

    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_restore_with_id_planner_storage_components(
        self, mock_dist_cp: MagicMock
    ) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        restore_options = RestoreOptions(restore_optimizers=False)
        DistributedCheckpointSaver.restore_with_id(
            checkpoint_id="path/to/snapshot",
            unit=my_unit,
            restore_options=restore_options,
            planner=DummyLoadPlanner(),
            storage_reader=DummyStorageReader(path="path/to/snapshot"),
        )
        planner = mock_dist_cp.load.call_args[1]["planner"]
        storage_reader = mock_dist_cp.load.call_args[1]["storage_reader"]

        self.assertIsInstance(planner, DummyLoadPlanner)
        self.assertIsInstance(storage_reader, DummyStorageReader)

    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_restore_allow_partial_loading(self, mock_dist_cp: MagicMock) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        restore_options = RestoreOptions(strict=False)
        DistributedCheckpointSaver.restore(
            path="path/to/snapshot",
            unit=my_unit,
            restore_options=restore_options,
        )

        allow_partial_load = mock_dist_cp.load.call_args[1][
            "planner"
        ].allow_partial_load
        self.assertTrue(allow_partial_load)

        restore_options = RestoreOptions(strict=True)
        DistributedCheckpointSaver.restore(
            path="path/to/snapshot",
            unit=my_unit,
            restore_options=restore_options,
        )

        allow_partial_load = mock_dist_cp.load.call_args[1][
            "planner"
        ].allow_partial_load
        self.assertFalse(allow_partial_load)

    @patch("torch.distributed.destroy_process_group")
    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_gloo_pg_restore(
        self, mock_dist_cp: MagicMock, mock_destroy_process_group: MagicMock
    ) -> None:
        my_unit = DummyAutoUnit(module=nn.Linear(2, 3))
        DistributedCheckpointSaver.restore(path="path/to/snapshot", unit=my_unit)
        process_group = mock_dist_cp.load.call_args.kwargs["process_group"]
        self.assertEqual(process_group, None)
        mock_destroy_process_group.assert_not_called()

    def test_save_restore_multi_optimizers(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 1

        my_unit = DummyMultiOptimUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        with tempfile.TemporaryDirectory() as temp_dir:
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                knob_options=KnobOptions(1),
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[dcp_cb])

            my_unit_clone = DummyMultiOptimUnit(input_dim=input_dim)
            dcp_cb.restore_from_latest(temp_dir, my_unit_clone)

    def test_save_restore_predict(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2

        my_unit = DummyPredictUnit(input_dim=input_dim)

        # pyre-ignore[16]: Add new attribute for testing
        my_unit.output_mean = DummyMeanMetric()

        # pyre-ignore[16]: Add at least one element to the metric
        my_unit.output_mean.update(1.0)

        dataloader = generate_dummy_stateful_dataloader(
            dataset_len, input_dim, batch_size
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                knob_options=KnobOptions(1),
                save_every_n_predict_steps=2,
            )

            predict(my_unit, dataloader, callbacks=[dcp_cb])

            generated_ckpts = os.listdir(temp_dir)
            expected_ckpts = [
                "epoch_0_predict_step_2",
                "epoch_0_predict_step_4",
                "epoch_1_predict_step_5",
            ]

            self.assertCountEqual(generated_ckpts, expected_ckpts)

            latest_ckpt_path = none_throws(get_latest_checkpoint_path(temp_dir))
            self.assertEqual(
                latest_ckpt_path, os.path.join(temp_dir, expected_ckpts[-1])
            )

            expected_keys = [
                "predict_progress",
                "predict_dataloader",
                "output_mean",
            ]

            # Check keys on a checkpoint other than the latest since it won't have dataloader state
            ckpt_path = f"{temp_dir}/{expected_ckpts[0]}"

            storage_reader = FsspecReader(ckpt_path)
            metadata = storage_reader.read_metadata()
            self.assertCountEqual(
                # Get base keys after the app_state wrapper
                {key.split(".")[1] for key in metadata.state_dict_metadata.keys()},
                expected_keys,
            )

            # Now make sure that the same exact keys are used when restoring
            with patch("torchtnt.framework.callbacks.dcp_saver.dcp.load") as mock_load:
                DistributedCheckpointSaver.restore(
                    ckpt_path, my_unit, predict_dataloader=dataloader
                )
                self.assertCountEqual(
                    [*mock_load.call_args[0][0]["app_state"].state_dict().keys()],
                    expected_keys,
                )

            # Double check that the module parameters are not overwritten when loading cktp
            my_unit = DummyPredictUnit(input_dim=input_dim)
            my_unit.module.weight.data.fill_(0.0)
            my_unit.module.bias.data.fill_(1.0)

            DistributedCheckpointSaver.restore(
                ckpt_path, my_unit, predict_dataloader=dataloader
            )

            self.assertTrue(
                torch.allclose(
                    my_unit.module.weight.data, torch.zeros(input_dim, input_dim)
                )
            )
            self.assertTrue(
                torch.allclose(
                    my_unit.module.bias.data, torch.ones(input_dim, input_dim)
                )
            )

    def test_save_restore_evaluate(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2

        my_unit = DummyEvalUnit(input_dim=input_dim)
        my_unit.loss = 0.1  # pyre-ignore[16]: Add new attribute for testing

        dataloader = generate_dummy_stateful_dataloader(
            dataset_len, input_dim, batch_size
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                knob_options=KnobOptions(1),
                save_every_n_eval_steps=2,
                best_checkpoint_config=BestCheckpointConfig(monitored_metric="loss"),
            )

            evaluate(my_unit, dataloader, callbacks=[dcp_cb])

            # For pure evaluation, the metric value should not be included
            generated_ckpts = os.listdir(temp_dir)
            expected_ckpts = [
                "epoch_0_eval_step_2",
                "epoch_0_eval_step_4",
            ]

            self.assertCountEqual(generated_ckpts, expected_ckpts)

            ckpt_path = none_throws(get_latest_checkpoint_path(temp_dir))
            self.assertEqual(ckpt_path, os.path.join(temp_dir, expected_ckpts[-1]))

            expected_keys = [
                "eval_progress",
                "eval_dataloader",
            ]
            storage_reader = FsspecReader(ckpt_path)
            metadata = storage_reader.read_metadata()
            self.assertCountEqual(
                # Get base keys after the app_state wrapper
                {key.split(".")[1] for key in metadata.state_dict_metadata.keys()},
                expected_keys,
            )

            # Now make sure that the same exact keys are used when restoring
            with patch("torchtnt.framework.callbacks.dcp_saver.dcp.load") as mock_load:
                DistributedCheckpointSaver.restore(
                    ckpt_path, my_unit, eval_dataloader=dataloader
                )
                self.assertCountEqual(
                    [*mock_load.call_args[0][0]["app_state"].state_dict().keys()],
                    expected_keys,
                )

            # Double check that the module parameters are not overwritten when loading cktp
            my_unit = DummyEvalUnit(input_dim=input_dim)
            my_unit.module.weight.data.fill_(0.0)
            my_unit.module.bias.data.fill_(1.0)

            DistributedCheckpointSaver.restore(
                ckpt_path, my_unit, predict_dataloader=dataloader
            )

            self.assertTrue(
                torch.allclose(
                    my_unit.module.weight.data, torch.zeros(input_dim, input_dim)
                )
            )
            self.assertTrue(
                torch.allclose(
                    my_unit.module.bias.data, torch.ones(input_dim, input_dim)
                )
            )

    def test_save_restore_fit_eval_every_n_epochs(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2

        my_unit = DummyAutoUnit(module=nn.Linear(input_dim, 2))
        my_unit.output_mean = DummyMeanMetric()
        my_unit.loss = 0.1

        train_dataloader = generate_dummy_stateful_dataloader(
            dataset_len, input_dim, batch_size
        )

        eval_dataloader = generate_dummy_stateful_dataloader(
            dataset_len, input_dim, batch_size
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                knob_options=KnobOptions(1),
                save_every_n_train_steps=2,
                save_every_n_eval_steps=2,
                best_checkpoint_config=BestCheckpointConfig(monitored_metric="loss"),
            )

            fit(
                my_unit,
                max_epochs=1,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                evaluate_every_n_epochs=1,
                callbacks=[dcp_cb],
            )

            generated_ckpts = os.listdir(temp_dir)
            # Since we are using FIT, the metric value should be included
            expected_ckpts_to_dl_mapping: Dict[str, Optional[str]] = {
                "epoch_0_train_step_2_eval_step_0_loss=0.1": "train_dataloader",
                "epoch_0_train_step_4_eval_step_0_loss=0.1": "train_dataloader",
                "epoch_1_train_step_5_eval_step_2_loss=0.1": "eval_dataloader",
                "epoch_1_train_step_5_eval_step_4_loss=0.1": "eval_dataloader",
                "epoch_1_train_step_5_eval_step_5_loss=0.1": None,
            }
            self.assertCountEqual(
                generated_ckpts, [*expected_ckpts_to_dl_mapping.keys()]
            )

            expected_keys = [
                "module",  # Both train and eval checkpoints save full app_state in fit
                "optimizer",
                "lr_scheduler",
                "train_progress",
                "eval_progress",
                "predict_progress",  # included because of AutoUnit
                "output_mean",
            ]

            for ckpt_path, dl_key in expected_ckpts_to_dl_mapping.items():
                full_ckpt_path = os.path.join(temp_dir, ckpt_path)
                expected_keys_with_dl = list(expected_keys)
                if dl_key:
                    expected_keys_with_dl.append(dl_key)

                storage_reader = FsspecReader(full_ckpt_path)
                metadata = storage_reader.read_metadata()
                self.assertCountEqual(
                    # Get base keys after the app_state wrapper
                    {key.split(".")[1] for key in metadata.state_dict_metadata.keys()},
                    expected_keys_with_dl,
                )

                # Now make sure that the same exact keys are used when restoring
                with patch(
                    "torchtnt.framework.callbacks.dcp_saver.dcp.load"
                ) as mock_load:
                    DistributedCheckpointSaver.restore(
                        full_ckpt_path,
                        my_unit,
                        train_dataloader=train_dataloader,
                        eval_dataloader=eval_dataloader,
                    )
                    self.assertCountEqual(
                        [*mock_load.call_args[0][0]["app_state"].state_dict().keys()],
                        expected_keys_with_dl,
                    )

    def test_save_restore_fit_save_every_n_eval_epochs(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2

        my_unit = DummyAutoUnit(module=nn.Linear(input_dim, 2))
        my_unit.output_mean = DummyMeanMetric()
        my_unit.loss = 0.1

        train_dataloader = generate_dummy_stateful_dataloader(
            dataset_len, input_dim, batch_size
        )

        eval_dataloader = generate_dummy_stateful_dataloader(
            dataset_len, input_dim, batch_size
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                knob_options=KnobOptions(1),
                save_every_n_eval_epochs=1,
                best_checkpoint_config=BestCheckpointConfig(monitored_metric="loss"),
            )

            fit(
                my_unit,
                max_epochs=1,
                evaluate_every_n_steps=1,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                callbacks=[dcp_cb],
            )

            generated_ckpts = os.listdir(temp_dir)
            # fbvscode.set_trace()
            # Since we are using FIT, the metric value should be included
            expected_ckpts_to_dl_mapping = {
                "epoch_0_train_step_1_eval_step_5_loss=0.1",
                "epoch_0_train_step_2_eval_step_10_loss=0.1",
                "epoch_0_train_step_3_eval_step_15_loss=0.1",
                "epoch_0_train_step_4_eval_step_20_loss=0.1",
                "epoch_0_train_step_5_eval_step_25_loss=0.1",
                "epoch_1_train_step_5_eval_step_30_loss=0.1",
            }
            self.assertCountEqual(generated_ckpts, [*expected_ckpts_to_dl_mapping])

            expected_keys = [
                "module",  # Both train and eval checkpoints save full app_state in fit
                "optimizer",
                "lr_scheduler",
                "train_progress",
                "eval_progress",
                "predict_progress",  # included because of AutoUnit
                "output_mean",
                "eval_dataloader",
                "train_dataloader",
            ]

            for ckpt_path in expected_ckpts_to_dl_mapping:
                full_ckpt_path = os.path.join(temp_dir, ckpt_path)
                expected_keys_with_dl = list(expected_keys)
                storage_reader = FsspecReader(full_ckpt_path)
                metadata = storage_reader.read_metadata()
                if ckpt_path == "epoch_1_train_step_5_eval_step_30_loss=0.1":
                    # remove dataloader keys as final checkpoint wont have them
                    expected_keys_with_dl = expected_keys_with_dl[:-1]
                appstate_keys = {
                    key.split(".")[1] for key in metadata.state_dict_metadata.keys()
                }
                self.assertCountEqual(
                    # Get base keys after the app_state wrapper
                    appstate_keys,
                    expected_keys_with_dl,
                    msg=f"key: {ckpt_path}, {expected_keys_with_dl=}, {appstate_keys=},",
                )

                # Now make sure that the same exact keys are used when restoring
                with patch(
                    "torchtnt.framework.callbacks.dcp_saver.dcp.load"
                ) as mock_load:
                    DistributedCheckpointSaver.restore(
                        full_ckpt_path,
                        my_unit,
                        train_dataloader=train_dataloader,
                        eval_dataloader=eval_dataloader,
                    )
                    self.assertCountEqual(
                        [*mock_load.call_args[0][0]["app_state"].state_dict().keys()],
                        expected_keys_with_dl,
                    )

    def test_save_fit_eval_every_n_steps(self) -> None:
        input_dim = 2

        my_unit = DummyAutoUnit(module=nn.Linear(input_dim, 2))
        my_unit.output_mean = DummyMeanMetric()

        train_dataloader = generate_dummy_stateful_dataloader(10, input_dim, 2)
        eval_dataloader = generate_dummy_stateful_dataloader(8, input_dim, 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                knob_options=KnobOptions(1),
                save_every_n_train_steps=2,
                save_every_n_eval_steps=2,
            )

            fit(
                my_unit,
                max_epochs=1,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                evaluate_every_n_steps=2,
                evaluate_every_n_epochs=None,
                callbacks=[dcp_cb],
            )

            generated_ckpts = os.listdir(temp_dir)
            expected_ckpts_to_dl_mapping: Dict[str, Tuple[str, ...]] = {
                # First train 2 steps
                "epoch_0_train_step_2_eval_step_0": ("train_dataloader",),
                # Then do a whole evaluation (4 steps)
                "epoch_0_train_step_2_eval_step_2": (
                    "train_dataloader",
                    "eval_dataloader",
                ),
                "epoch_0_train_step_2_eval_step_4": (
                    "train_dataloader",
                    "eval_dataloader",
                ),
                # Then train other two steps
                "epoch_0_train_step_4_eval_step_4": ("train_dataloader",),
                # Finally do a whole evaluation (4 steps)
                "epoch_0_train_step_4_eval_step_6": (
                    "train_dataloader",
                    "eval_dataloader",
                ),
                "epoch_0_train_step_4_eval_step_8": (
                    "train_dataloader",
                    "eval_dataloader",
                ),
                # Last checkpoint (on_train_end)
                "epoch_1_train_step_5_eval_step_8": (),
            }
            self.assertCountEqual(
                generated_ckpts, [*expected_ckpts_to_dl_mapping.keys()]
            )

            expected_keys = [
                "module",  # Both train and eval checkpoints save full app_state in fit
                "optimizer",
                "lr_scheduler",
                "train_progress",
                "eval_progress",
                "predict_progress",  # included because of AutoUnit
                "output_mean",
            ]

            for ckpt_path, expected_dls in expected_ckpts_to_dl_mapping.items():
                expected_keys_with_dls = [*expected_keys, *expected_dls]
                full_ckpt_path = os.path.join(temp_dir, ckpt_path)
                storage_reader = FsspecReader(full_ckpt_path)
                metadata = storage_reader.read_metadata()
                self.assertCountEqual(
                    # Get base keys after the app_state wrapper
                    {key.split(".")[1] for key in metadata.state_dict_metadata.keys()},
                    expected_keys_with_dls,
                )

                # Now make sure that the same exact keys are used when restoring
                with patch(
                    "torchtnt.framework.callbacks.dcp_saver.dcp.load"
                ) as mock_load:
                    DistributedCheckpointSaver.restore(
                        full_ckpt_path,
                        my_unit,
                        train_dataloader=train_dataloader,
                        eval_dataloader=eval_dataloader,
                    )
                    self.assertCountEqual(
                        [*mock_load.call_args[0][0]["app_state"].state_dict().keys()],
                        expected_keys_with_dls,
                    )


class DummyStatefulDataLoader:
    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self.state_dict_call_count = 0
        self.load_state_dict_call_count = 0

    def state_dict(self) -> Dict[str, Any]:
        self.state_dict_call_count += 1
        return {"some_data": 1}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.load_state_dict_call_count += 1
        return None

    def __iter__(self) -> Iterator[object]:
        return iter(self.dataloader)


class DummySavePlanner(DefaultSavePlanner):
    def __init__(self) -> None:
        super().__init__()

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        storage_meta: Optional[StorageMeta],
        is_coordinator: bool,
    ) -> None:
        super().set_up_planner(state_dict, storage_meta, is_coordinator)


class DummyLoadPlanner(DefaultLoadPlanner):
    def __init__(self) -> None:
        super().__init__()

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Optional[Metadata],
        is_coordinator: bool,
    ) -> None:
        super().set_up_planner(state_dict, metadata, is_coordinator)


class DummyStorageWriter(FileSystemWriter):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def set_up_storage_writer(
        self, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        pass


class DummyStorageReader(FileSystemReader):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def set_up_storage_reader(
        self, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        pass
