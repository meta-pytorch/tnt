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
from typing import Any, Dict, Iterator, List
from unittest import mock
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torchsnapshot.test_utils import assert_state_dict_eq, check_state_dict_eq
from torchtnt.framework._test_utils import (
    DummyAutoUnit,
    DummyTrainUnit,
    generate_random_dataloader,
    get_dummy_train_state,
)
from torchtnt.framework.callbacks.checkpointer_types import KnobOptions, RestoreOptions
from torchtnt.framework.callbacks.torchsnapshot_saver import (
    _exclude_progress_from_replicated,
    _override_knobs,
    TorchSnapshotSaver,
)
from torchtnt.framework.train import train
from torchtnt.utils.distributed import get_global_rank, spawn_multi_process
from torchtnt.utils.env import seed
from torchtnt.utils.test_utils import skip_if_not_distributed


class TorchSnapshotSaverTest(unittest.TestCase):
    def tearDown(self) -> None:
        # needed for github test, to reset WORLD_SIZE env var
        os.environ.pop("WORLD_SIZE", None)

    def test_save_restore(self) -> None:
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
            snapshot_cb = TorchSnapshotSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[snapshot_cb])

            end_num_steps_completed = my_unit.train_progress.num_steps_completed
            self.assertGreater(len(expected_paths), 0)
            snapshot_cb.restore(expected_paths[0], my_unit)
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
            snapshot_cb = TorchSnapshotSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
            )
            train(
                my_unit,
                stateful_dataloader,
                max_steps=max_steps,
                callbacks=[snapshot_cb],
            )
            # state_dict has been called once on dataloader
            self.assertEqual(stateful_dataloader.state_dict_call_count, 1)
            self.assertEqual(stateful_dataloader.load_state_dict_call_count, 0)

            # restoring from first checkpoint, has dataloader in manifest
            snapshot_cb.restore(
                temp_dir + f"/epoch_{0}_train_step_{save_every_n_train_steps}",
                my_unit,
                train_dataloader=stateful_dataloader,
            )
            # load_state_dict has been called once on dataloader
            self.assertEqual(stateful_dataloader.load_state_dict_call_count, 1)

            # restoring from last checkpoint (on train end), does not have dataloader state in manifest

            with self.assertLogs(level="WARNING") as log:
                snapshot_cb.restore(
                    temp_dir + f"/epoch_{1}_train_step_{max_steps}",
                    my_unit,
                    train_dataloader=stateful_dataloader,
                )
                # load_state_dict is not called again on dataloader because there is no dataloader in manifest
                self.assertEqual(stateful_dataloader.load_state_dict_call_count, 1)
                self.assertEqual(
                    log.output[0],
                    "WARNING:torchtnt.utils.rank_zero_log:train_dataloader was passed to `restore` but no train dataloader exists in the Snapshot",
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
            snapshot_cb = TorchSnapshotSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[snapshot_cb])

            with mock.patch(
                "torchtnt.framework.callbacks.torchsnapshot_saver.TorchSnapshotSaver.restore"
            ) as mock_restore:
                restored = snapshot_cb.restore_from_latest(temp_dir, my_unit)
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
            snapshot_cb = TorchSnapshotSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[snapshot_cb])

            end_num_steps_completed = my_unit.train_progress.num_steps_completed
            self.assertGreater(len(expected_paths), 0)
            snapshot_cb.restore(
                expected_paths[0],
                my_unit,
                restore_options=RestoreOptions(restore_train_progress=False),
            )
            restored_num_steps_completed = my_unit.train_progress.num_steps_completed
            # no train progress was restored so the progress after restoration should be the same as the progress before restoration
            self.assertEqual(restored_num_steps_completed, end_num_steps_completed)

    @patch("torchtnt.framework.callbacks.torchsnapshot_saver.torchsnapshot")
    def test_save_restore_no_optimizer_restore(
        self, mock_torchsnapshot: MagicMock
    ) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        restore_options = RestoreOptions(restore_optimizers=False)
        TorchSnapshotSaver.restore(
            path="path/to/snapshot", unit=my_unit, restore_options=restore_options
        )
        app_state = mock_torchsnapshot.Snapshot().restore.call_args.args[0]
        self.assertNotIn("optimizer", app_state)
        TorchSnapshotSaver.restore(path="path/to/snapshot", unit=my_unit)
        app_state = mock_torchsnapshot.Snapshot().restore.call_args.args[0]
        self.assertIn("optimizer", app_state)

    @patch("torchtnt.framework.callbacks.torchsnapshot_saver.torchsnapshot")
    def test_save_restore_no_lr_scheduler_restore(
        self, mock_torchsnapshot: MagicMock
    ) -> None:
        my_unit = DummyAutoUnit(module=nn.Linear(2, 3))
        restore_options = RestoreOptions(restore_lr_schedulers=False)
        TorchSnapshotSaver.restore(
            path="path/to/snapshot", unit=my_unit, restore_options=restore_options
        )
        app_state = mock_torchsnapshot.Snapshot().restore.call_args.args[0]
        self.assertNotIn("lr_scheduler", app_state)
        TorchSnapshotSaver.restore(path="path/to/snapshot", unit=my_unit)
        app_state = mock_torchsnapshot.Snapshot().restore.call_args.args[0]
        self.assertIn("lr_scheduler", app_state)

    def test_restore_strict(self) -> None:
        my_unit = DummyAutoUnit(module=torch.nn.Linear(2, 2))
        with tempfile.TemporaryDirectory() as temp_dir:
            state = get_dummy_train_state()
            snapshot_cb = TorchSnapshotSaver(
                temp_dir, save_every_n_train_steps=1, async_checkpoint=False
            )
            snapshot_cb.on_train_step_end(state, my_unit)

            # add a new parameter to the module
            my_unit.module2 = torch.nn.Linear(2, 2)

            with self.assertRaisesRegex(
                AssertionError,
                "module2 is absent in both manifest and flattened.",
            ):
                TorchSnapshotSaver.restore(
                    path=os.path.join(temp_dir, "epoch_0_train_step_0"),
                    unit=my_unit,
                    strict=True,
                )

            with self.assertLogs(level="WARNING") as log:
                TorchSnapshotSaver.restore(
                    path=os.path.join(temp_dir, "epoch_0_train_step_0"),
                    unit=my_unit,
                    strict=False,
                )
                self.assertEqual(
                    log.output[0],
                    "WARNING:torchtnt.utils.rank_zero_log:module2 was passed to `restore` but does not exists in the snapshot",
                )

    @skip_if_not_distributed
    def test_save_restore_ddp(self) -> None:
        spawn_multi_process(
            2,
            "gloo",
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

        snapshot_cb = TorchSnapshotSaver(
            temp_dir,
            save_every_n_epochs=save_every_n_epochs,
            replicated=["**"],
        )
        temp_dir = snapshot_cb.dirpath
        train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[snapshot_cb])
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
            snapshot_cb.restore(ckpt_path, my_new_unit)

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

    def test_knob_override(self) -> None:
        env_var = "TORCHSNAPSHOT_MAX_PER_RANK_IO_CONCURRENCY_OVERRIDE"
        knob_options = KnobOptions(max_per_rank_io_concurrency=1)
        with _override_knobs(knob_options):
            self.assertEqual(os.environ[env_var], str(1))
        self.assertNotIn(env_var, os.environ)

        with _override_knobs(KnobOptions(max_per_rank_io_concurrency=None)):
            self.assertNotIn(env_var, os.environ)

    @patch("torchtnt.framework.callbacks.torchsnapshot_saver.torchsnapshot")
    def test_sync_checkpoint(self, _: MagicMock) -> None:
        """
        Tests the _sync_snapshot function is called if async is turned off.
        """
        my_unit = DummyTrainUnit(input_dim=2)
        state = get_dummy_train_state()

        snapshot_cb = TorchSnapshotSaver(
            "tmp/foo",
            save_every_n_train_steps=1,
            async_checkpoint=False,
        )
        snapshot_cb._sync_snapshot = MagicMock()
        snapshot_cb.on_train_step_end(state, my_unit)
        snapshot_cb._sync_snapshot.assert_called_once()

    def test_exclude_progress_from_replicated(self) -> None:
        """
        Tests that replicated is populated correctly with progress excluded
        """

        module = nn.Linear(2, 3)
        my_unit = DummyAutoUnit(module=module)
        keys = my_unit.app_state().keys()

        progress_keys = {"train_progress", "eval_progress", "predict_progress"}

        replicated = _exclude_progress_from_replicated(my_unit.app_state())
        for key in keys:
            if key not in progress_keys:
                self.assertIn(f"{key}/**", replicated)

        # since we exclude 3 keys (train, eval, predict)
        self.assertEqual(len(keys) - 3, len(replicated))

        # check that progress is not included
        for progress_key in progress_keys:
            self.assertNotIn(f"{progress_key}/", replicated)

    @patch("torchtnt.framework.callbacks.torchsnapshot_saver.Snapshot.take")
    def test_exclude_progress_from_replicated_e2e(self, mock_take: MagicMock) -> None:
        """
        Tests that replicated is populated correctly during snapshotting
        """

        module = nn.Linear(2, 3)
        my_unit = DummyAutoUnit(module=module)
        state = get_dummy_train_state()

        with tempfile.TemporaryDirectory() as temp_dir:
            for replicated_value in (None, ["optimizer/**"], ["**"]):
                tss = TorchSnapshotSaver(
                    dirpath=temp_dir,
                    save_every_n_train_steps=1,
                    async_checkpoint=False,
                    replicated=replicated_value,
                )

                progress_keys = {"train_progress", "eval_progress", "predict_progress"}

                tss.on_train_step_end(state, my_unit)
                replicated = mock_take.call_args.kwargs["replicated"]

                if replicated_value is None:
                    self.assertEqual(replicated, [])
                elif replicated_value == ["optimizer/**"]:
                    self.assertEqual(replicated, ["optimizer/**"])
                elif replicated_value == ["**"]:
                    expected_replicated = [
                        f"{key}/**"
                        for key in my_unit.app_state().keys()
                        if key not in progress_keys
                    ]
                    # this is added outside of the unit's app_state so it should be included
                    expected_replicated.append("rng_state/**")

                    self.assertEqual(set(replicated), set(expected_replicated))


class DummyStatefulDataLoader:
    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self.state_dict_call_count = 0
        self.load_state_dict_call_count = 0

    def state_dict(self) -> Dict[str, Any]:
        self.state_dict_call_count += 1
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        self.load_state_dict_call_count += 1
        return None

    def __iter__(self) -> Iterator[object]:
        return iter(self.dataloader)
