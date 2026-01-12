#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import math
import os
import shutil
import tempfile
import time
import unittest
from typing import Any, cast, Iterable, List, Optional
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from torch import nn
from torchtnt.framework._test_utils import (
    Batch,
    DummyAutoUnit,
    DummyFitUnit,
    DummyPredictUnit,
    DummyTrainUnit,
    generate_random_dataloader,
    get_dummy_fit_state,
    get_dummy_train_state,
)
from torchtnt.framework.callbacks.base_checkpointer import (
    BaseCheckpointer as BaseCheckpointer,
)
from torchtnt.framework.callbacks.checkpointer_types import RestoreOptions
from torchtnt.framework.callbacks.lambda_callback import Lambda
from torchtnt.framework.evaluate import evaluate
from torchtnt.framework.fit import fit
from torchtnt.framework.predict import predict
from torchtnt.framework.state import ActivePhase, State
from torchtnt.framework.train import train
from torchtnt.framework.unit import (
    AppStateMixin,
    TEvalData,
    TPredictData,
    TrainUnit,
    TTrainData,
    TTrainUnit,
)
from torchtnt.utils.checkpoint import BestCheckpointConfig, get_latest_checkpoint_path
from torchtnt.utils.distributed import get_global_rank, spawn_multi_process
from torchtnt.utils.env import init_from_env
from torchtnt.utils.test_utils import skip_if_not_distributed


class BaseCheckpointSaver(BaseCheckpointer):
    """
    A basic checkpointer class that generates an empty directory upon checkpoint
    """

    def __init__(
        self,
        dirpath: str,
        *,
        save_every_n_train_steps: Optional[int] = None,
        save_every_n_epochs: Optional[int] = None,
        save_every_n_eval_steps: Optional[int] = None,
        save_every_n_eval_epochs: Optional[int] = None,
        save_every_n_predict_steps: Optional[int] = None,
        keep_last_n_checkpoints: Optional[int] = None,
        best_checkpoint_config: Optional[BestCheckpointConfig] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__(
            dirpath,
            save_every_n_train_steps=save_every_n_train_steps,
            save_every_n_epochs=save_every_n_epochs,
            save_every_n_eval_steps=save_every_n_eval_steps,
            save_every_n_eval_epochs=save_every_n_eval_epochs,
            save_every_n_predict_steps=save_every_n_predict_steps,
            keep_last_n_checkpoints=keep_last_n_checkpoints,
            best_checkpoint_config=best_checkpoint_config,
            process_group=process_group,
        )
        self._latest_checkpoint_path: str = ""

    def _checkpoint_impl(
        self, state: State, unit: AppStateMixin, checkpoint_id: str, hook: str
    ) -> bool:
        self._latest_checkpoint_path = checkpoint_id
        if not os.path.exists(checkpoint_id):
            os.mkdir(checkpoint_id)
        return True

    @staticmethod
    def restore(
        path: str,
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        eval_dataloader: Optional[Iterable[TEvalData]] = None,
        predict_dataloader: Optional[Iterable[TPredictData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
        msg: str = "",
        restored_checkpoint_path: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        if restored_checkpoint_path is not None:
            if len(restored_checkpoint_path):
                restored_checkpoint_path[0] = path
            else:
                restored_checkpoint_path.append(path)
        print(f"Checkpoint restored with message: {msg}")
        return


class BaseCheckpointerTest(unittest.TestCase):
    cuda_available: bool = torch.cuda.is_available()
    distributed_available: bool = torch.distributed.is_available()

    def tearDown(self) -> None:
        # needed for github test, to reset WORLD_SIZE env var
        os.environ.pop("WORLD_SIZE", None)

    def test_save_every_n_train_steps(self) -> None:
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
            checkpointer = BaseCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
            )
            train(
                my_unit,
                dataloader,
                max_epochs=max_epochs,
                callbacks=[checkpointer],
            )
            for path in expected_paths:
                self.assertTrue(os.path.exists(path) and os.path.isdir(path))

    def test_save_every_n_train_epochs(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 3
        expected_steps_per_epoch = math.ceil(dataset_len / batch_size)
        save_every_n_train_epochs = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        with tempfile.TemporaryDirectory() as temp_dir:
            expected_path = os.path.join(
                temp_dir,
                f"epoch_{save_every_n_train_epochs}_train_step_{expected_steps_per_epoch * (save_every_n_train_epochs)}",
            )
            checkpointer = BaseCheckpointSaver(
                temp_dir,
                save_every_n_epochs=save_every_n_train_epochs,
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[checkpointer])
            self.assertTrue(
                os.path.exists(expected_path) and os.path.isdir(expected_path)
            )

    def test_save_every_n_eval_epochs(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 3
        expected_steps_per_epoch = math.ceil(dataset_len / batch_size)
        save_every_n_eval_epochs = 2

        my_unit = DummyFitUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        with tempfile.TemporaryDirectory() as temp_dir:
            expected_path = os.path.join(
                temp_dir,
                f"epoch_2_train_step_{expected_steps_per_epoch * 2}_eval_step_{expected_steps_per_epoch * 2}",  # 2 eval epochs
            )
            checkpointer = BaseCheckpointSaver(
                temp_dir,
                save_every_n_eval_epochs=save_every_n_eval_epochs,
            )
            fit(
                my_unit,
                dataloader,
                eval_dataloader=dataloader,
                evaluate_every_n_epochs=1,
                max_epochs=max_epochs,
                callbacks=[checkpointer],
            )
            self.assertTrue(
                os.path.exists(expected_path) and os.path.isdir(expected_path)
            )

    def test_save_fit_entrypoint(self) -> None:
        input_dim = 2

        my_unit = DummyFitUnit(input_dim=input_dim)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpointer = BaseCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=1,
                save_every_n_epochs=1,
                save_every_n_eval_epochs=1,
            )
            train_state = get_dummy_train_state()
            fit_state = get_dummy_fit_state()
            my_unit.train_progress._num_steps_completed = 15
            my_unit.eval_progress._num_steps_completed = 10

            checkpointer.on_train_step_end(train_state, my_unit)
            self.assertIn(
                f"epoch_0_train_step_{15}", checkpointer._latest_checkpoint_path
            )

            checkpointer.on_train_step_end(fit_state, my_unit)
            self.assertIn(
                f"epoch_0_train_step_{15}_eval_step_{10}",
                checkpointer._latest_checkpoint_path,
            )

            checkpointer.on_train_epoch_end(train_state, my_unit)
            self.assertIn(
                f"epoch_0_train_step_{15}", checkpointer._latest_checkpoint_path
            )

            checkpointer.on_train_epoch_end(fit_state, my_unit)
            self.assertIn(
                f"epoch_0_train_step_{15}_eval_step_{10}",
                checkpointer._latest_checkpoint_path,
            )

            fit_state._active_phase = ActivePhase.EVALUATE
            checkpointer.on_eval_epoch_end(fit_state, my_unit)
            self.assertIn(
                f"epoch_0_train_step_{15}_eval_step_{10}",
                checkpointer._latest_checkpoint_path,
            )

    @patch.object(BaseCheckpointSaver, "_checkpoint_impl")
    def test_save_eval_entrypoint(self, mock_checkpoint_impl: MagicMock) -> None:
        my_unit = DummyFitUnit(input_dim=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpointer = BaseCheckpointSaver(
                temp_dir,
                save_every_n_eval_steps=2,
                best_checkpoint_config=BestCheckpointConfig(
                    monitored_metric="val_loss", mode="min"
                ),
                keep_last_n_checkpoints=1,
            )

            ckpt_container: List[str] = []

            def _checkpoint_impl_side_effect(
                state: State, unit: AppStateMixin, checkpoint_id: str, hook: str
            ) -> bool:
                ckpt_container.append(checkpoint_id)
                return True

            mock_checkpoint_impl.side_effect = _checkpoint_impl_side_effect

            eval_dataloader = generate_random_dataloader(10, 2, 1)

            warning_container: List[str] = []
            with patch(
                "torchtnt.framework.callbacks.base_checkpointer.logging.Logger.warning",
                side_effect=warning_container.append,
            ):
                evaluate(my_unit, eval_dataloader, callbacks=[checkpointer])

            # Verify that checkpoint optimality tracking was disabled
            self.assertIn(
                "Disabling best_checkpoint_config, since it is not supported for eval or predict entrypoints.",
                warning_container,
            )
            self.assertIn(
                "Disabling keep_last_n_checkpoints, since is not supported for eval or predict entrypoints.",
                warning_container,
            )

            # Make sure that the correct checkpoints were saved, without tracked metrics
            expected_ckpts = [
                f"{temp_dir}/epoch_0_eval_step_{i*2}" for i in range(1, 6)
            ]
            self.assertEqual(ckpt_container, expected_ckpts)

    @patch.object(BaseCheckpointSaver, "_checkpoint_impl")
    def test_save_predict_entrypoint(self, mock_checkpoint_impl: MagicMock) -> None:
        my_unit = DummyPredictUnit(input_dim=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpointer = BaseCheckpointSaver(
                temp_dir,
                save_every_n_predict_steps=1,
            )

            ckpt_container: List[str] = []

            def _checkpoint_impl_side_effect(
                state: State, unit: AppStateMixin, checkpoint_id: str, hook: str
            ) -> bool:
                ckpt_container.append(checkpoint_id)
                return True

            mock_checkpoint_impl.side_effect = _checkpoint_impl_side_effect

            predict_dataloader = generate_random_dataloader(10, 2, 1)

            predict(my_unit, predict_dataloader, callbacks=[checkpointer])

            # Make sure that the correct checkpoints were saved
            expected_ckpts = [
                f"{temp_dir}/epoch_0_predict_step_{i}" for i in range(1, 11)
            ]

            expected_ckpts.append(
                f"{temp_dir}/epoch_1_predict_step_10"
            )  # We always expect checkpoint on predict end

            self.assertEqual(ckpt_container, expected_ckpts)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_restore_from_latest(self, mock_stdout: MagicMock) -> None:
        my_unit = DummyTrainUnit(input_dim=2)

        # Using only phase-aware checkpoints
        with tempfile.TemporaryDirectory() as temp_dir:
            # create a dummy directory structure
            os.makedirs(os.path.join(temp_dir, "epoch_0_train_step_0"))
            os.makedirs(os.path.join(temp_dir, "epoch_0_eval_step_4_acc=26.9"))
            os.makedirs(os.path.join(temp_dir, "epoch_1_train_step_2"))
            os.makedirs(os.path.join(temp_dir, "epoch_1_eval_step_1_acc=5.0"))

            restored_checkpoint_container = []
            BaseCheckpointSaver.restore_from_latest(
                temp_dir,
                my_unit,
                msg="foo",
                restored_checkpoint_path=restored_checkpoint_container,
            )

            # make sure correct checkpoint was restored
            self.assertEqual(
                restored_checkpoint_container[0],
                os.path.join(temp_dir, "epoch_1_eval_step_1_acc=5.0"),
            )

            # ensure **kwargs are plumbed through correctly
            self.assertEqual(
                mock_stdout.getvalue(), "Checkpoint restored with message: foo\n"
            )

        # Using only phase-naive checkpoints
        with tempfile.TemporaryDirectory() as temp_dir:
            # create a dummy directory structure
            os.makedirs(os.path.join(temp_dir, "epoch_0_step_0"))
            os.makedirs(os.path.join(temp_dir, "epoch_0_step_4_acc=26.9"))
            os.makedirs(os.path.join(temp_dir, "epoch_1_step_2"))
            os.makedirs(os.path.join(temp_dir, "epoch_1_step_1_acc=5.0"))

            restored_checkpoint_container = []
            BaseCheckpointSaver.restore_from_latest(
                temp_dir,
                my_unit,
                restored_checkpoint_path=restored_checkpoint_container,
            )

            # make sure correct checkpoint was restored
            self.assertEqual(
                restored_checkpoint_container[0],
                os.path.join(temp_dir, "epoch_1_step_2"),
            )

        # Using a mix of phase-naive and phase-aware checkpoints
        with tempfile.TemporaryDirectory() as temp_dir:
            # create a dummy directory structure
            os.makedirs(os.path.join(temp_dir, "epoch_0_eval_step_0"))
            os.makedirs(os.path.join(temp_dir, "epoch_0_step_4_acc=26.9"))
            os.makedirs(os.path.join(temp_dir, "epoch_0_step_3"))
            os.makedirs(os.path.join(temp_dir, "epoch_0_train_step_1_acc=5.0"))

            restored_checkpoint_container = []
            BaseCheckpointSaver.restore_from_latest(
                temp_dir,
                my_unit,
                restored_checkpoint_path=restored_checkpoint_container,
            )

            # make sure correct checkpoint was restored
            self.assertEqual(
                restored_checkpoint_container[0],
                os.path.join(temp_dir, "epoch_0_step_4_acc=26.9"),
            )

            # add a larger phase naive checkpoint
            os.makedirs(os.path.join(temp_dir, "epoch_0_step_5"))
            BaseCheckpointSaver.restore_from_latest(
                temp_dir,
                my_unit,
                restored_checkpoint_path=restored_checkpoint_container,
            )
            self.assertEqual(
                restored_checkpoint_container[0],
                os.path.join(temp_dir, "epoch_0_step_5"),
            )

    def test_restore_from_latest_empty_dir(self) -> None:
        input_dim = 2
        save_every_n_train_steps = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        with tempfile.TemporaryDirectory() as temp_dir:
            bcs_cb = BaseCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
            )

            with self.assertLogs(level="WARNING") as log:
                restored = bcs_cb.restore_from_latest(temp_dir, my_unit)
                self.assertEqual(
                    log.output,
                    [
                        f"WARNING:torchtnt.utils.checkpoint:Input dirpath doesn't contain any subdirectories: {temp_dir}"
                    ],
                )
                self.assertFalse(restored)

    def test_restore_from_best(self) -> None:
        input_dim = 2
        state = get_dummy_train_state()

        with tempfile.TemporaryDirectory() as temp_dir:
            bcs_cb = BaseCheckpointSaver(temp_dir)

            my_unit = DummyTrainUnit(input_dim=input_dim)
            bcs_cb._generate_checkpoint_and_upkeep(state, my_unit, hook="foo")
            os.rename(
                os.path.join(temp_dir, "epoch_0_train_step_0"),
                os.path.join(temp_dir, "epoch_0_train_step_0_val_loss=0.01"),
            )

            my_unit.train_progress._num_steps_completed = 10
            bcs_cb._generate_checkpoint_and_upkeep(state, my_unit, hook="foo")
            os.rename(
                os.path.join(temp_dir, "epoch_0_train_step_10"),
                os.path.join(temp_dir, "epoch_0_train_step_10_val_loss=-0.1"),
            )

            my_unit.train_progress._num_steps_completed = 20
            bcs_cb._generate_checkpoint_and_upkeep(state, my_unit, hook="foo")
            os.rename(
                os.path.join(temp_dir, "epoch_0_train_step_20"),
                os.path.join(temp_dir, "epoch_0_train_step_20_val_loss=0.1"),
            )

            my_unit = DummyTrainUnit(input_dim=input_dim)
            with self.assertLogs(level="INFO") as log:
                restored = bcs_cb.restore_from_best(
                    temp_dir, my_unit, "val_loss", "min"
                )
                self.assertTrue(restored)
                self.assertIn(
                    f"INFO:torchtnt.utils.rank_zero_log:Loading checkpoint from {os.path.join(temp_dir, 'epoch_0_train_step_10_val_loss=-0.1')}",
                    log.output,
                )

            my_unit = DummyTrainUnit(input_dim=input_dim)
            with self.assertLogs(level="INFO") as log:
                restored = bcs_cb.restore_from_best(
                    temp_dir, my_unit, "val_loss", "max"
                )
                self.assertTrue(restored)
                self.assertIn(
                    f"INFO:torchtnt.utils.rank_zero_log:Loading checkpoint from {os.path.join(temp_dir, 'epoch_0_train_step_20_val_loss=0.1')}",
                    log.output,
                )

            # Adding phase-naive checkpoints to the mix
            my_unit.train_progress._num_steps_completed = 5
            bcs_cb._generate_checkpoint_and_upkeep(state, my_unit, hook="foo")
            os.rename(
                os.path.join(temp_dir, "epoch_0_train_step_5"),
                os.path.join(
                    temp_dir, "epoch_0_step_5_val_loss=-5.0"
                ),  # should be selected on min
            )

            restored_checkpoint_container = []
            bcs_cb.restore_from_best(
                temp_dir,
                my_unit,
                "val_loss",
                "min",
                restored_checkpoint_path=restored_checkpoint_container,
            )

            # make sure correct checkpoint was restored
            self.assertEqual(
                restored_checkpoint_container[0],
                os.path.join(temp_dir, "epoch_0_step_5_val_loss=-5.0"),
            )

            restored_checkpoint_container = []
            bcs_cb.restore_from_best(
                temp_dir,
                my_unit,
                "val_loss",
                "max",
                restored_checkpoint_path=restored_checkpoint_container,
            )

            # make sure correct checkpoint was restored
            self.assertEqual(
                restored_checkpoint_container[0],
                os.path.join(temp_dir, "epoch_0_train_step_20_val_loss=0.1"),
            )

    def test_restore_from_best_empty_dir(self) -> None:
        input_dim = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        with tempfile.TemporaryDirectory() as temp_dir:
            bcs_cb = BaseCheckpointSaver(
                temp_dir,
            )

            with self.assertLogs(level="WARNING") as log:
                restored = bcs_cb.restore_from_best(
                    temp_dir, my_unit, "val_loss", "min"
                )
                self.assertIn(
                    f"WARNING:torchtnt.framework.callbacks.base_checkpointer:No checkpoints with metric name val_loss were found in {temp_dir}. Not loading any checkpoint.",
                    log.output,
                )
                self.assertFalse(restored)

    def test_save_on_train_end(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2

        expected_path = (
            f"epoch_{max_epochs}_train_step_{max_epochs * (dataset_len // batch_size)}"
        )

        my_unit = DummyTrainUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertFalse(os.path.exists(os.path.join(temp_dir, expected_path)))
            checkpoint_cb = BaseCheckpointSaver(
                temp_dir,
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[checkpoint_cb])

            expected_path = f"epoch_{max_epochs}_train_step_{max_epochs * (dataset_len // batch_size)}"
            self.assertTrue(os.path.exists(os.path.join(temp_dir, expected_path)))

            with self.assertLogs(level="WARNING") as log:
                checkpoint_cb._checkpoint_manager._metadata_fnames = [".metadata"]
                # create metadata file
                with open(os.path.join(temp_dir, expected_path, ".metadata"), "w"):
                    pass

                # train again without resetting progress
                train(
                    my_unit,
                    dataloader,
                    max_epochs=max_epochs,
                    callbacks=[checkpoint_cb],
                )
                self.assertEqual(
                    log.output,
                    [
                        "WARNING:torchtnt.framework.callbacks.base_checkpointer:Final checkpoint already exists, skipping."
                    ],
                )

    def test_save_on_train_end_on_fit(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 6

        for save_every_n_eval_epochs, expected_last_ckpt in [
            (None, "epoch_6_train_step_30_eval_step_25"),
            (2, "epoch_6_train_step_30_eval_step_30"),
        ]:
            my_unit = DummyAutoUnit(module=nn.Linear(input_dim, 2))
            train_dataloader = generate_random_dataloader(
                dataset_len, input_dim, batch_size
            )
            eval_dataloader = generate_random_dataloader(
                dataset_len, input_dim, batch_size
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_cb = BaseCheckpointSaver(
                    temp_dir,
                    save_every_n_epochs=2,
                    save_every_n_eval_epochs=save_every_n_eval_epochs,
                )
                fit(
                    my_unit,
                    train_dataloader=train_dataloader,
                    eval_dataloader=eval_dataloader,
                    max_epochs=max_epochs,
                    evaluate_every_n_epochs=1,
                    callbacks=[checkpoint_cb],
                )
                expected_path = os.path.join(temp_dir, expected_last_ckpt)
                self.assertTrue(os.path.exists(expected_path))
                self.assertEqual(
                    checkpoint_cb._checkpoint_manager._ckpt_paths[-1].path,
                    expected_path,
                )

    @skip_if_not_distributed
    def test_directory_sync_collective(self) -> None:
        spawn_multi_process(
            2,
            "gloo",
            self._directory_sync_collective,
        )

    @staticmethod
    def _directory_sync_collective() -> None:
        init_from_env()
        try:
            if get_global_rank() == 0:
                temp_dir = tempfile.mkdtemp()
            else:
                temp_dir = "foo"

            bcs = BaseCheckpointSaver(temp_dir)
            dirpath = bcs.dirpath
            tc = unittest.TestCase()
            tc.assertTrue("tmp" in dirpath)
            tc.assertFalse("foo" in dirpath)
        finally:
            dist.barrier()  # avoid race condition
            if get_global_rank() == 0:
                shutil.rmtree(temp_dir)  # delete temp directory

    def test_invalid_args(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                ValueError, "Invalid value passed for save_every_n_train_steps.*"
            ):
                BaseCheckpointSaver(temp_dir, save_every_n_train_steps=-2)
            with self.assertRaisesRegex(
                ValueError, "Invalid value passed for save_every_n_train_steps.*"
            ):
                BaseCheckpointSaver(temp_dir, save_every_n_train_steps=0)
            with self.assertRaisesRegex(
                ValueError, "Invalid value passed for save_every_n_epochs.*"
            ):
                BaseCheckpointSaver(temp_dir, save_every_n_epochs=-2)
            with self.assertRaisesRegex(
                ValueError, "Invalid value passed for save_every_n_epochs.*"
            ):
                BaseCheckpointSaver(temp_dir, save_every_n_epochs=0)

    @skip_if_not_distributed
    def test_process_group_plumbing(self) -> None:
        spawn_multi_process(
            2,
            "gloo",
            self._test_process_group_plumbing_gloo,
        )
        spawn_multi_process(
            2,
            "gloo",  # inner test mocks nccl backend
            self._test_process_group_plumbing_nccl,
        )

    @staticmethod
    def _test_process_group_plumbing_gloo() -> None:
        checkpoint_cb = BaseCheckpointSaver(
            "foo",
            process_group=None,
        )
        tc = unittest.TestCase()
        tc.assertEqual(
            dist.get_backend(checkpoint_cb._process_group), dist.Backend.GLOO
        )
        if dist.get_backend(dist.group.WORLD) == dist.Backend.GLOO:
            # verify no new process group was created
            tc.assertEqual(checkpoint_cb._process_group, dist.group.WORLD)

    @staticmethod
    @patch("torch.cuda.nccl.version", return_value=(1, 0, 0))
    def _test_process_group_plumbing_nccl(_: MagicMock) -> None:
        with patch("torch.distributed.get_backend", return_value=dist.Backend.NCCL):
            checkpoint_cb = BaseCheckpointSaver(
                "foo",
                process_group=None,
            )

        tc = unittest.TestCase()
        tc.assertIsNotNone(checkpoint_cb._process_group)
        tc.assertEqual(
            checkpoint_cb._process_group._get_backend_name(), dist.Backend.GLOO
        )
        # check that a new process group was created
        tc.assertNotEqual(checkpoint_cb._process_group, dist.group.WORLD)

    def test_keep_last_n_checkpoints_e2e(self) -> None:
        """
        Tests removing checkpoint directories e2e
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        with tempfile.TemporaryDirectory() as temp_dir:
            bc = BaseCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=2,
                keep_last_n_checkpoints=1,
            )
            # Artificially increase the step duration, otherwise torchcheckpoint
            # doesn't have the time to save all checkpoints and will skip some.
            slowdown = Lambda(on_train_step_end=lambda *_: time.sleep(0.1))

            train(
                my_unit,
                dataloader,
                max_epochs=max_epochs,
                callbacks=[bc, slowdown],
            )
            dirs = os.listdir(temp_dir)
            self.assertEqual(len(dirs), 1)
            self.assertIn(
                f"epoch_{max_epochs}_train_step_{dataset_len // batch_size * max_epochs}",
                os.listdir(temp_dir),
            )

    def test_best_checkpoint_attr_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bcs = BaseCheckpointSaver(
                temp_dir,
                save_every_n_epochs=1,
                best_checkpoint_config=BestCheckpointConfig(
                    monitored_metric="train_loss",
                    mode="min",
                ),
            )

            state = get_dummy_train_state()
            my_val_unit = MyValLossUnit()

            error_container = []
            with patch(
                "torchtnt.framework.callbacks.base_checkpointer.logging.Logger.error",
                side_effect=error_container.append,
            ):
                bcs.on_train_epoch_end(state, my_val_unit)

            self.assertIn(
                "Unit does not have attribute train_loss, unable to retrieve metric to checkpoint. Will not be included in checkpoint path, nor tracked for optimality.",
                error_container,
            )

            self.assertTrue(os.path.exists(f"{temp_dir}/epoch_0_train_step_0"))

    def test_best_checkpoint_no_top_k(self) -> None:
        """
        Tests basic functionality of best checkpoint saving

        - Checks that the best checkpoint is saved when the monitored metric
        - top_k is not configured, so no checkpoints should be deleted
        """

        for mode in ("min", "max"):
            with tempfile.TemporaryDirectory() as temp_dir:
                bcs = BaseCheckpointSaver(
                    temp_dir,
                    save_every_n_epochs=1,
                    best_checkpoint_config=BestCheckpointConfig(
                        monitored_metric="train_loss",
                        # pyre-fixme: Incompatible parameter type [6]
                        mode=mode,
                    ),
                )

                state = get_dummy_train_state()
                my_train_unit = MyTrainLossUnit()

                my_train_unit.train_loss = None
                bcs.on_train_epoch_end(state, my_train_unit)
                # none metric-value will not be updated in checkpoint dirpaths
                self.assertEqual(bcs._checkpoint_manager._ckpt_paths, [])
                self.assertEqual(os.listdir(temp_dir), ["epoch_0_train_step_0"])

                my_train_unit.train_loss = 0.01
                bcs.on_train_epoch_end(state, my_train_unit)
                self.assertEqual(
                    [str(x) for x in bcs._checkpoint_manager._ckpt_paths],
                    [os.path.join(temp_dir, "epoch_0_train_step_0_train_loss=0.01")],
                )

                my_train_unit.train_loss = 0.02
                my_train_unit.train_progress.increment_epoch()
                bcs.on_train_epoch_end(state, my_train_unit)
                self.assertEqual(
                    [str(x) for x in bcs._checkpoint_manager._ckpt_paths],
                    (
                        [
                            os.path.join(
                                temp_dir, "epoch_1_train_step_0_train_loss=0.02"
                            ),
                            os.path.join(
                                temp_dir, "epoch_0_train_step_0_train_loss=0.01"
                            ),
                        ]
                        if mode == "min"
                        else [
                            os.path.join(
                                temp_dir, "epoch_0_train_step_0_train_loss=0.01"
                            ),
                            os.path.join(
                                temp_dir, "epoch_1_train_step_0_train_loss=0.02"
                            ),
                        ]
                    ),
                )

                my_train_unit.train_loss = 0.015
                my_train_unit.train_progress.increment_epoch()
                bcs.on_train_epoch_end(state, my_train_unit)
                self.assertEqual(
                    [str(x) for x in bcs._checkpoint_manager._ckpt_paths],
                    (
                        [
                            os.path.join(
                                temp_dir, "epoch_1_train_step_0_train_loss=0.02"
                            ),
                            os.path.join(
                                temp_dir, "epoch_2_train_step_0_train_loss=0.015"
                            ),
                            os.path.join(
                                temp_dir, "epoch_0_train_step_0_train_loss=0.01"
                            ),
                        ]
                        if mode == "min"
                        else [
                            os.path.join(
                                temp_dir, "epoch_0_train_step_0_train_loss=0.01"
                            ),
                            os.path.join(
                                temp_dir, "epoch_2_train_step_0_train_loss=0.015"
                            ),
                            os.path.join(
                                temp_dir, "epoch_1_train_step_0_train_loss=0.02"
                            ),
                        ]
                    ),
                )

    def test_best_checkpoint_top_k(self) -> None:
        # test top_k = 1
        with tempfile.TemporaryDirectory() as temp_dir:
            bcs = BaseCheckpointSaver(
                temp_dir,
                save_every_n_epochs=1,
                best_checkpoint_config=BestCheckpointConfig(
                    monitored_metric="train_loss",
                    mode="min",
                ),
                keep_last_n_checkpoints=1,
            )

            state = get_dummy_train_state()
            my_train_unit = MyTrainLossUnit()

            bcs.on_train_epoch_end(state, my_train_unit)
            self.assertEqual(
                [str(x) for x in bcs._checkpoint_manager._ckpt_paths],
                [os.path.join(temp_dir, "epoch_0_train_step_0_train_loss=0.01")],
            )

            my_train_unit.train_loss = 0.02
            my_train_unit.train_progress.increment_epoch()
            bcs.on_train_epoch_end(state, my_train_unit)
            self.assertEqual(
                [str(x) for x in bcs._checkpoint_manager._ckpt_paths],
                [
                    os.path.join(temp_dir, "epoch_0_train_step_0_train_loss=0.01"),
                ],
            )

            my_train_unit.train_loss = 0.001
            my_train_unit.train_progress.increment_epoch()
            bcs.on_train_epoch_end(state, my_train_unit)
            self.assertEqual(
                [str(x) for x in bcs._checkpoint_manager._ckpt_paths],
                [
                    os.path.join(temp_dir, "epoch_2_train_step_0_train_loss=0.001"),
                ],
            )

        # test top_k = 2
        with tempfile.TemporaryDirectory() as temp_dir:
            bcs = BaseCheckpointSaver(
                temp_dir,
                save_every_n_epochs=1,
                best_checkpoint_config=BestCheckpointConfig(
                    monitored_metric="train_loss",
                    mode="min",
                ),
                keep_last_n_checkpoints=2,
            )

            state = get_dummy_train_state()
            my_train_unit = MyTrainLossUnit()

            bcs.on_train_epoch_end(state, my_train_unit)
            self.assertEqual(
                [str(x) for x in bcs._checkpoint_manager._ckpt_paths],
                [os.path.join(temp_dir, "epoch_0_train_step_0_train_loss=0.01")],
            )

            my_train_unit.train_loss = 0.02
            my_train_unit.train_progress.increment_epoch()
            bcs.on_train_epoch_end(state, my_train_unit)
            self.assertEqual(
                [str(x) for x in bcs._checkpoint_manager._ckpt_paths],
                [
                    os.path.join(temp_dir, "epoch_1_train_step_0_train_loss=0.02"),
                    os.path.join(temp_dir, "epoch_0_train_step_0_train_loss=0.01"),
                ],
            )

            my_train_unit.train_loss = 0.001
            my_train_unit.train_progress.increment_epoch()
            bcs.on_train_epoch_end(state, my_train_unit)
            self.assertEqual(
                [str(x) for x in bcs._checkpoint_manager._ckpt_paths],
                [
                    os.path.join(temp_dir, "epoch_0_train_step_0_train_loss=0.01"),
                    os.path.join(temp_dir, "epoch_2_train_step_0_train_loss=0.001"),
                ],
            )

    def test_no_assert_error_in_on_train_end(self) -> None:
        """
        Tests no assertion is thrown when using BestCheckpointConfig in on_train_end
        """

        input_dim = 2
        dataset_len = 4
        batch_size = 2
        max_epochs = 2

        expected_path = (
            f"epoch_{max_epochs}_step_{max_epochs * (dataset_len // batch_size)}"
        )

        my_unit = MyValLossUnit()
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertFalse(os.path.exists(os.path.join(temp_dir, expected_path)))
            checkpoint_cb = BaseCheckpointSaver(
                temp_dir,
                best_checkpoint_config=BestCheckpointConfig("val_loss", "min"),
                keep_last_n_checkpoints=2,
                save_every_n_train_steps=1,
            )
            train(
                my_unit,
                dataloader,
                max_epochs=max_epochs,
                callbacks=[checkpoint_cb],
            )

    def test_get_tracked_metric_value(self) -> None:
        """
        Tests that _get_tracked_metric_value returns the correct value
        """
        val_loss_unit = MyValLossUnit()

        val_loss_ckpt_cb = BaseCheckpointSaver(
            dirpath="checkpoint",
            best_checkpoint_config=BestCheckpointConfig("val_loss", "min"),
        )
        val_loss = val_loss_ckpt_cb._get_tracked_metric_value(val_loss_unit)
        self.assertEqual(0.01, val_loss)

        # pyre-ignore
        val_loss_unit.val_loss = "0.01"  # Test when returned as a string
        val_loss_from_s = val_loss_ckpt_cb._get_tracked_metric_value(val_loss_unit)
        self.assertEqual(0.01, val_loss_from_s)

        # pyre-ignore
        val_loss_unit.val_loss = "hola"  # Test weird metric value
        error_container = []
        with patch(
            "torchtnt.framework.callbacks.base_checkpointer.logging.Logger.error",
            side_effect=error_container.append,
        ):
            val_loss = val_loss_ckpt_cb._get_tracked_metric_value(val_loss_unit)

        self.assertIn(
            "Unable to convert monitored metric val_loss to a float: could not convert string to float: 'hola'. "
            "Please ensure the value can be converted to float and is not a multi-element tensor value. Will not be "
            "included in checkpoint path, nor tracked for optimality.",
            error_container,
        )

        val_loss_unit.val_loss = float("nan")  # Test nan metric value
        error_container = []
        with patch(
            "torchtnt.framework.callbacks.base_checkpointer.logging.Logger.error",
            side_effect=error_container.append,
        ):
            val_loss = val_loss_ckpt_cb._get_tracked_metric_value(val_loss_unit)

        self.assertEqual(
            [
                "Monitored metric 'val_loss' is NaN. Will not be included in checkpoint path, nor tracked for optimality."
            ],
            error_container,
        )
        self.assertIsNone(val_loss)

        # test infinite metric values
        for inf_val in [float("inf"), -float("inf")]:
            val_loss_unit.val_loss = inf_val
            error_container = []
            with patch(
                "torchtnt.framework.callbacks.base_checkpointer.logging.Logger.error",
                side_effect=error_container.append,
            ):
                val_loss = val_loss_ckpt_cb._get_tracked_metric_value(val_loss_unit)

            self.assertIn(
                "Monitored metric 'val_loss' is inf. Will not be included in checkpoint path, nor tracked for optimality.",
                error_container,
            )

        # test with mismatched monitored metric
        train_loss_ckpt_cb = BaseCheckpointSaver(
            dirpath="checkpoint",
            best_checkpoint_config=BestCheckpointConfig("train_loss", "max"),
        )
        error_container = []
        with patch(
            "torchtnt.framework.callbacks.base_checkpointer.logging.Logger.error",
            side_effect=error_container.append,
        ):
            val_loss = train_loss_ckpt_cb._get_tracked_metric_value(val_loss_unit)

        self.assertIn(
            "Unit does not have attribute train_loss, unable to retrieve metric to checkpoint. "
            "Will not be included in checkpoint path, nor tracked for optimality.",
            error_container,
        )

        ckpt_cb = BaseCheckpointSaver(
            dirpath="checkpoint",
        )
        no_metric = ckpt_cb._get_tracked_metric_value(val_loss_unit)
        self.assertIsNone(no_metric)

    def test_multi_phase_e2e(self) -> None:
        """
        Test several checkpoint functionalities working as a whole
        """
        train_dl = generate_random_dataloader(10, 2, 1)
        eval_dl = generate_random_dataloader(5, 2, 1)
        unit = DummyAutoUnit(module=nn.Linear(2, 2))

        with tempfile.TemporaryDirectory() as temp_dir:
            # Include some legacy formats to verify backwards compatibility
            legacy_ckpts = [
                f"{temp_dir}/epoch_0_step_0",
                f"{temp_dir}/epoch_0_step_5",
                f"{temp_dir}/epoch_0_step_10_eval_loss=0.01",
            ]
            for path in legacy_ckpts:
                os.mkdir(path)

            cb = BaseCheckpointSaver(
                dirpath=temp_dir,
                save_every_n_epochs=2,
                save_every_n_eval_epochs=1,
            )
            self.assertEqual(
                get_latest_checkpoint_path(temp_dir),
                f"{temp_dir}/epoch_0_step_10_eval_loss=0.01",
            )

            # Do fit to generate multi-phase checkpoints
            fit(
                unit,
                train_dataloader=train_dl,
                eval_dataloader=eval_dl,
                max_epochs=5,
                max_train_steps_per_epoch=5,
                max_eval_steps_per_epoch=2,
                callbacks=[cb],
            )

            multi_phase_ckpts = [
                f"{temp_dir}/epoch_1_train_step_5_eval_step_2",
                f"{temp_dir}/epoch_2_train_step_10_eval_step_2",
                f"{temp_dir}/epoch_2_train_step_10_eval_step_4",
                f"{temp_dir}/epoch_3_train_step_15_eval_step_6",
                f"{temp_dir}/epoch_4_train_step_20_eval_step_6",
                f"{temp_dir}/epoch_4_train_step_20_eval_step_8",
                f"{temp_dir}/epoch_5_train_step_25_eval_step_10",
            ]
            for ckpt in multi_phase_ckpts:
                self.assertTrue(os.path.exists(ckpt))

            self.assertEqual(
                get_latest_checkpoint_path(temp_dir),
                f"{temp_dir}/epoch_5_train_step_25_eval_step_10",
            )

            # train to generate single-phase checkpoints
            train(
                cast(TTrainUnit, unit),
                train_dataloader=train_dl,
                max_epochs=10,  # had already trained 5 epochs
                max_steps_per_epoch=5,
                callbacks=[cb],
            )

            train_ckpts = [
                f"{temp_dir}/epoch_6_train_step_30",  # cktp every 2 steps
                f"{temp_dir}/epoch_8_train_step_40",
                f"{temp_dir}/epoch_10_train_step_50",
            ]
            for ckpt in train_ckpts:
                self.assertTrue(os.path.exists(ckpt))

            self.assertEqual(
                get_latest_checkpoint_path(temp_dir),
                f"{temp_dir}/epoch_10_train_step_50",
            )

            # Add a mid recency phase-naive checkpoint
            mid_phase_naive = f"{temp_dir}/epoch_5_step_36"
            os.mkdir(mid_phase_naive)

            # Make sure that all the checkpoints are actually parsed and sorted
            new_cb = BaseCheckpointSaver(dirpath=temp_dir, keep_last_n_checkpoints=1)
            all_paths = (
                legacy_ckpts + multi_phase_ckpts + [mid_phase_naive] + train_ckpts
            )
            self.assertEqual(
                all_paths, [str(x) for x in new_cb._checkpoint_manager._ckpt_paths]
            )

            # Add one extra checkpoint to verify metric parsing
            metric_ckpt = (
                f"{temp_dir}/epoch_12_train_step_60_eval_step_24_eval_loss=0.1"
            )
            os.mkdir(metric_ckpt)

            metric_cb = BaseCheckpointSaver(
                dirpath=temp_dir,
                keep_last_n_checkpoints=2,
                best_checkpoint_config=BestCheckpointConfig("eval_loss", "min"),
            )
            min_optim = [
                f"{temp_dir}/epoch_12_train_step_60_eval_step_24_eval_loss=0.1",
                f"{temp_dir}/epoch_0_step_10_eval_loss=0.01",
            ]
            self.assertEqual(
                min_optim, [str(x) for x in metric_cb._checkpoint_manager._ckpt_paths]
            )

    @skip_if_not_distributed
    def test_directory_path_synced(self) -> None:
        spawn_multi_process(
            2,
            "gloo",
            self._test_directory_path_synced,
        )

    @staticmethod
    def _test_directory_path_synced() -> None:
        init_from_env()
        tc = unittest.TestCase()

        temp_dir = tempfile.mkdtemp() if get_global_rank() == 0 else ""
        bcs = BaseCheckpointSaver(
            temp_dir,
            save_every_n_epochs=1,
        )

        try:
            state = get_dummy_train_state()
            my_train_unit = MyTrainLossUnit()

            if dist.get_rank() == 0:
                my_train_unit.train_progress._num_epochs_completed = 10
            else:
                my_train_unit.train_progress._num_epochs_completed = 3

            bcs.on_train_epoch_end(state, my_train_unit)
            tc.assertEqual(len(bcs._checkpoint_manager._ckpt_paths), 1)
            tc.assertEqual(
                str(bcs._checkpoint_manager._ckpt_paths[0]),
                os.path.join(bcs.dirpath, "epoch_10_train_step_0"),
            )
            tc.assertEqual(
                os.listdir(bcs.dirpath),
                ["epoch_10_train_step_0"],
            )
        finally:
            dist.barrier()  # avoid race condition
            if get_global_rank() == 0:
                shutil.rmtree(temp_dir)  # delete temp directory

    def test_dist_not_initialized(self) -> None:
        """
        Tests that BaseCheckpointSaver cannot be initialized without dist being initialized
        if world size > 1
        """
        os.environ["WORLD_SIZE"] = "2"
        with self.assertRaisesRegex(
            RuntimeError, "Running in a distributed environment"
        ):
            BaseCheckpointSaver("foo")


class MyValLossUnit(TrainUnit[Batch]):
    def __init__(self) -> None:
        super().__init__()
        self.val_loss = 0.01

    def train_step(self, state: State, data: Batch) -> None:
        return None


class MyTrainLossUnit(TrainUnit[Batch]):
    def __init__(self) -> None:
        super().__init__()
        self.train_loss: Optional[float] = 0.01

    def train_step(self, state: State, data: Batch) -> None:
        return None
