# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import os
import pickle
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from torch import nn
from torchsnapshot import Snapshot
from torchsnapshot.snapshot import SNAPSHOT_METADATA_FNAME
from torchtnt.framework._test_utils import Batch
from torchtnt.framework.state import State
from torchtnt.framework.unit import TrainUnit
from torchtnt.utils import get_global_rank, init_from_env
from torchtnt.utils.checkpoint import (
    _metadata_exists,
    _retrieve_checkpoint_dirpaths,
    BestCheckpointConfig,
    CheckpointManager,
    CheckpointPath,
    does_checkpoint_exist,
    get_best_checkpoint_path,
    get_checkpoint_dirpaths,
    get_latest_checkpoint_path,
    MetricData,
    Phase,
)
from torchtnt.utils.distributed import (
    PGWrapper,
    rank_zero_read_and_broadcast,
    spawn_multi_process,
)
from torchtnt.utils.fsspec import get_filesystem
from torchtnt.utils.test_utils import skip_if_not_distributed

METADATA_FNAME: str = ".metadata"


class CheckpointPathTest(unittest.TestCase):

    def test_create_checkpoint_path(self) -> None:
        # phase-naive and metric-naive
        ckpt = CheckpointPath("foo", epoch=0, step=1)
        self.assertEqual(ckpt.path, "foo/epoch_0_step_1")

        # phase-aware and metric-naive
        ckpt = CheckpointPath("foo", epoch=0, step={Phase.TRAIN: 1})
        self.assertEqual(ckpt.path, "foo/epoch_0_train_step_1")

        # phase-aware and metric-aware
        ckpt = CheckpointPath(
            "foo",
            epoch=0,
            step={Phase.TRAIN: 1, Phase.EVALUATE: 1},
            metric_data=MetricData("foo", 1.0),
        )
        self.assertEqual(ckpt.path, "foo/epoch_0_train_step_1_eval_step_1_foo=1.0")

        # evaluation only
        ckpt = CheckpointPath(
            "foo",
            epoch=0,
            step={Phase.EVALUATE: 1},
        )
        self.assertEqual(ckpt.path, "foo/epoch_0_eval_step_1")

        # prediction only
        ckpt = CheckpointPath(
            "foo",
            epoch=0,
            step={Phase.PREDICT: 1},
        )
        self.assertEqual(ckpt.path, "foo/epoch_0_predict_step_1")

        # all phases - not expected but should work
        ckpt = CheckpointPath(
            "foo",
            epoch=0,
            step={Phase.TRAIN: 1, Phase.EVALUATE: 1, Phase.PREDICT: 1},
        )
        self.assertEqual(
            ckpt.path, "foo/epoch_0_train_step_1_eval_step_1_predict_step_1"
        )

        # nan metric value
        with self.assertRaisesRegex(
            ValueError,
            "Value of monitored metric 'foo' can't be NaN in CheckpointPath.",
        ):
            CheckpointPath(
                "foo",
                epoch=0,
                step={Phase.TRAIN: 1, Phase.EVALUATE: 1},
                metric_data=MetricData("foo", float("nan")),
            )

        # inf metric value
        with self.assertRaisesRegex(
            ValueError,
            "Value of monitored metric 'foo' can't be inf in CheckpointPath.",
        ):
            CheckpointPath(
                "foo",
                epoch=0,
                step={Phase.TRAIN: 1, Phase.EVALUATE: 1},
                metric_data=MetricData("foo", float("inf")),
            )

    def test_from_str(self) -> None:
        # invalid paths
        malformed_paths = [
            "foo/step_20",
            "foo/epoch_50",
            "epoch_30",
            "foo/epoch_20_step",
            "foo/epoch_20_step_30_val_loss=1a",
            "foo/epoch_2_step_15_mean=hello",
            "foo/epoch_2.6_step_23",
            "foo/epoch_3_pred_step_3",
            "foo/epoch_3__step_3",
            "foo/epoch_2_predict_step_2_eval_step_1",
            "foo/epoch_2_predict_step_3.2",
        ]
        for path in malformed_paths:
            with self.assertRaisesRegex(
                ValueError, f"Attempted to parse malformed checkpoint path: {path}"
            ):
                CheckpointPath.from_str(path)

        # valid paths
        valid_paths = [
            ("foo/epoch_0_step_1", CheckpointPath("foo", epoch=0, step=1)),
            (
                "foo_bar/fizz_buzz/epoch_0_step_1",
                CheckpointPath("foo_bar/fizz_buzz", epoch=0, step={Phase.NONE: 1}),
            ),
            (
                "foo/epoch_14_step_3_mean=15.0",
                CheckpointPath(
                    "foo", epoch=14, step=3, metric_data=MetricData("mean", 15.0)
                ),
            ),
            (
                "foo/epoch_14_step_3_train_loss=15.0",
                CheckpointPath(
                    "foo",
                    epoch=14,
                    step={Phase.NONE: 3},
                    metric_data=MetricData("train_loss", 15.0),
                ),
            ),
            (
                "foo/epoch_14_step_3_loss=-27.35",
                CheckpointPath(
                    "foo", epoch=14, step=3, metric_data=MetricData("loss", -27.35)
                ),
            ),
            (
                "/foo/epoch_14_step_3_loss=-27.35",
                CheckpointPath(
                    "/foo", epoch=14, step=3, metric_data=MetricData("loss", -27.35)
                ),
            ),
            (
                "foo/epoch_2_eval_step_23",
                CheckpointPath("foo", epoch=2, step={Phase.EVALUATE: 23}),
            ),
            (
                "foo/epoch_14_predict_step_5",
                CheckpointPath("foo", epoch=14, step={Phase.PREDICT: 5}),
            ),
            (
                "foo/epoch_14_train_step_3_eval_loss=0.1",
                CheckpointPath(
                    "foo",
                    epoch=14,
                    step={Phase.TRAIN: 3},
                    metric_data=MetricData("eval_loss", 0.1),
                ),
            ),
            (
                "foo/bar/epoch_23_step_31_mean_loss_squared=0.0",
                CheckpointPath(
                    "foo/bar/",
                    epoch=23,
                    step=31,
                    metric_data=MetricData("mean_loss_squared", 0.0),
                ),
            ),
            (
                "file://some/path/checkpoints/0b20e70f-9ad2-4904-b7d6-e8da48087d61/epoch_2_step_1_acc=0.98",
                CheckpointPath(
                    "file://some/path/checkpoints/0b20e70f-9ad2-4904-b7d6-e8da48087d61",
                    epoch=2,
                    step={Phase.NONE: 1},
                    metric_data=MetricData("acc", 0.98),
                ),
            ),
            (
                "foo/bar/epoch_23_train_step_31_mean_loss_squared=0.0",
                CheckpointPath(
                    "foo/bar/",
                    epoch=23,
                    step={Phase.TRAIN: 31},
                    metric_data=MetricData("mean_loss_squared", 0.0),
                ),
            ),
            (
                "file://path/some/checkpoints_dir/0b20e70f-9ad2-4904-b7d6-e8da48087d61/epoch_2_eval_step_1_acc=0.98",
                CheckpointPath(
                    "file://path/some/checkpoints_dir/0b20e70f-9ad2-4904-b7d6-e8da48087d61",
                    epoch=2,
                    step={Phase.EVALUATE: 1},
                    metric_data=MetricData("acc", 0.98),
                ),
            ),
            (
                "foo/bar/epoch_23_train_step_31_eval_step_15_mean_loss_squared=-23.6",
                CheckpointPath(
                    "foo/bar/",
                    epoch=23,
                    step={Phase.TRAIN: 31, Phase.EVALUATE: 15},
                    metric_data=MetricData("mean_loss_squared", -23.6),
                ),
            ),
            (
                "file://path/some/checkpoints_dir/0b20e70f-9ad2-4904-b7d6-e8da48087d61/epoch_56_train_step_60_eval_step_1",
                CheckpointPath(
                    "file://path/some/checkpoints_dir/0b20e70f-9ad2-4904-b7d6-e8da48087d61",
                    epoch=56,
                    step={Phase.TRAIN: 60, Phase.EVALUATE: 1},
                ),
            ),
            (
                "foo/bar/epoch_0_train_step_0_eval_step_0_mean_loss_squared=0.0",
                CheckpointPath(
                    "foo/bar/",
                    epoch=0,
                    step={Phase.TRAIN: 0, Phase.EVALUATE: 0},
                    metric_data=MetricData("mean_loss_squared", 0.0),
                ),
            ),
            (
                "foo/bar/epoch_1_train_step_2_eval_step_3_eval_loss=6.486097566010406e+18",
                CheckpointPath(
                    "foo/bar/",
                    epoch=1,
                    step={Phase.TRAIN: 2, Phase.EVALUATE: 3},
                    metric_data=MetricData("eval_loss", 6.486097566010406e18),
                ),
            ),
        ]
        for path, expected_ckpt in valid_paths:
            parsed_ckpt = CheckpointPath.from_str(path)
            self.assertEqual(parsed_ckpt, expected_ckpt)
            self.assertEqual(parsed_ckpt.path, path)

        # with a trailing slash
        ckpt = CheckpointPath.from_str("foo/epoch_0_step_1/")
        self.assertEqual(ckpt, CheckpointPath("foo", epoch=0, step=1))
        self.assertEqual(ckpt.path, "foo/epoch_0_step_1")

    def test_compare_by_recency(self) -> None:
        old = CheckpointPath("foo", epoch=0, step=1)
        new = CheckpointPath("foo", epoch=1, step=1)
        self.assertTrue(new.newer_than(old))
        self.assertFalse(old.newer_than(new))
        self.assertFalse(new == old)

        old = CheckpointPath("foo", epoch=3, step=5)
        new = CheckpointPath("foo", epoch=3, step=9)
        self.assertTrue(new.newer_than(old))
        self.assertFalse(old.newer_than(new))
        self.assertFalse(new == old)

        twin1 = CheckpointPath(
            "foo", epoch=2, step=5, metric_data=MetricData("foo", 1.0)
        )
        almost_twin = CheckpointPath(
            "foo", epoch=2, step=5, metric_data=MetricData("bar", 2.0)
        )

        self.assertFalse(twin1.newer_than(almost_twin))
        self.assertFalse(almost_twin.newer_than(twin1))
        self.assertFalse(twin1 == almost_twin)

        twin2 = CheckpointPath(
            "foo", epoch=2, step=5, metric_data=MetricData("foo", 1.0)
        )
        self.assertTrue(twin1 == twin2)

        legacy1 = CheckpointPath("foo", epoch=3, step=3)
        old = CheckpointPath("foo", epoch=3, step={Phase.TRAIN: 5})
        legacy2 = CheckpointPath("foo", epoch=3, step=7)
        new = CheckpointPath("foo", epoch=3, step={Phase.TRAIN: 5, Phase.EVALUATE: 5})
        legacy3 = CheckpointPath("foo", epoch=3, step=10)
        legacy4 = CheckpointPath("foor", epoch=4, step=12)

        self.assertTrue(old.newer_than(legacy1))
        self.assertTrue(new.newer_than(legacy1))
        self.assertTrue(new.newer_than(old))
        self.assertFalse(old.newer_than(new))
        self.assertTrue(legacy2.newer_than(old))
        self.assertTrue(new.newer_than(legacy2))
        self.assertTrue(new.newer_than(legacy3))
        self.assertFalse(legacy3.newer_than(new))
        self.assertTrue(legacy4.newer_than(legacy3))
        self.assertFalse(new == old)

        old = CheckpointPath("foo", epoch=3, step={Phase.TRAIN: 5})
        new = CheckpointPath("foo", epoch=3, step={Phase.TRAIN: 6})
        self.assertTrue(new.newer_than(old))
        self.assertFalse(old.newer_than(new))

        train_only = CheckpointPath("foo", epoch=3, step={Phase.TRAIN: 10})
        eval_only = CheckpointPath("foo", epoch=3, step={Phase.EVALUATE: 10})
        multiphase_1 = CheckpointPath(
            "foo", epoch=3, step={Phase.TRAIN: 5, Phase.EVALUATE: 5}
        )
        multiphase_2 = CheckpointPath(
            "foo", epoch=4, step={Phase.TRAIN: 15, Phase.EVALUATE: 10}
        )
        multiphase_3 = CheckpointPath(
            "foo", epoch=4, step={Phase.TRAIN: 20, Phase.EVALUATE: 10}
        )

        self.assertTrue(eval_only > train_only)
        self.assertFalse(eval_only < train_only)
        self.assertTrue(train_only < multiphase_1)
        self.assertTrue(eval_only > multiphase_1)
        self.assertTrue(eval_only < multiphase_2)
        self.assertTrue(multiphase_2 < multiphase_3)

        predict_1 = CheckpointPath("foo", epoch=3, step={Phase.PREDICT: 10})
        predict_2 = CheckpointPath("foo", epoch=4, step={Phase.PREDICT: 10})
        predict_3 = CheckpointPath("foo", epoch=4, step={Phase.PREDICT: 20})
        self.assertTrue(predict_1 < predict_2)
        self.assertTrue(predict_2 < predict_3)

    def test_compare_by_optimality(self) -> None:
        ckpt1 = CheckpointPath(
            "foo", epoch=1, step=1, metric_data=MetricData("bar", 1.0)
        )

        # tracking different metrics
        ckpt2 = CheckpointPath(
            "foo", epoch=1, step=1, metric_data=MetricData("baz", 1.0)
        )
        with self.assertRaisesRegex(
            AssertionError,
            "Attempted to compare optimality of checkpoints tracking different metrics",
        ):
            ckpt1.more_optimal_than(ckpt2, mode="min")

        smaller = CheckpointPath(
            "foo", epoch=0, step=1, metric_data=MetricData("foo", 1.0)
        )
        larger = CheckpointPath(
            "foo",
            epoch=0,
            step={Phase.TRAIN: 1},
            metric_data=MetricData("foo", 2.0),
        )
        self.assertTrue(larger.more_optimal_than(smaller, mode="max"))
        self.assertFalse(smaller.more_optimal_than(larger, mode="max"))
        self.assertTrue(smaller.more_optimal_than(larger, mode="min"))
        self.assertFalse(larger.more_optimal_than(smaller, mode="min"))

    def test_compare_by_optimality_nan(self) -> None:
        ckpt1 = CheckpointPath("foo", epoch=0, step=1)
        ckpt2 = CheckpointPath(
            "foo", epoch=9, step=2, metric_data=MetricData("foo", 1.0)
        )
        self.assertTrue(ckpt2.more_optimal_than(ckpt1, mode="min"))
        self.assertFalse(ckpt1.more_optimal_than(ckpt2, mode="min"))

        ckpt3 = CheckpointPath("foo", epoch=0, step=2)
        self.assertTrue(ckpt3.more_optimal_than(ckpt1, mode="min"))
        self.assertFalse(ckpt1.more_optimal_than(ckpt3, mode="min"))

    def test_pickling(self) -> None:
        for path in (
            "foo/epoch_0_step_1",
            "file://some/path/checkpoints/0b20e70f-9ad2-4904-b7d6-e8da48087d61/epoch_2_step_1_acc=0.98",
            "foo/epoch_0_train_step_2_eval_step_5",
        ):
            ckpt = CheckpointPath.from_str(path)

            pickled = pickle.dumps(ckpt)

            # Don't test equality because of custom protocol
            self.assertTrue(path in str(pickled))

            unpickled = pickle.loads(pickled)
            self.assertEqual(unpickled, ckpt)

    def test_checkpoint_ordering(self) -> None:
        """
        Tests the sort utilities
        """
        paths = [
            "foo/epoch_1_step_20",
            "foo/epoch_4_step_130",
            "foo/epoch_0_step_10_val_loss=10.0",
        ]
        ckpts = [CheckpointPath.from_str(x) for x in paths]
        sorted_paths = [str(x) for x in sorted(ckpts)]
        self.assertEqual(sorted_paths, [paths[2], paths[0], paths[1]])

        paths = [
            "foo/epoch_0_step_1",
            "foo/epoch_1_train_step_20_val_loss=0.09",
            "foo/epoch_0_train_step_10_val_loss=10.0",
            "foo/epoch_1_step_30_val_loss=29.0",
            "foo/epoch_0_eval_step_2_val_loss=13.0",
            "foo/epoch_1_train_step_25_val_loss=0.06",
        ]
        ckpts = [CheckpointPath.from_str(path) for path in paths]
        self.assertEqual(
            [str(path) for path in sorted(ckpts)],
            [paths[0], paths[2], paths[4], paths[1], paths[5], paths[3]],
        )
        paths = [
            "foo/epoch_1_train_step_20_eval_step_10_val_loss=0.09",
            "foo/epoch_0_train_step_10_eval_step_0_val_loss=10.0",
            "foo/epoch_1_step_32_val_loss=29.0",  # phase naive
            "foo/epoch_1_train_step_20_eval_step_15_val_loss=0.02",
            "foo/epoch_0_train_step_15_eval_step_0_val_loss=13.0",
            "foo/epoch_1_train_step_25_val_loss=0.06",
            "foo/epoch_0_train_step_15_eval_step_5_val_loss=18.0",
            "foo/epoch_0_eval_step_25_val_loss=18.0",
            "foo/epoch_0_step_10",  # phase naive
        ]
        ckpts = [CheckpointPath.from_str(path) for path in paths]
        for ckpt in ckpts:
            print(ckpt.__repr__())

        self.assertEqual(
            [str(path) for path in sorted(ckpts)],
            [
                paths[8],
                paths[1],
                paths[4],
                paths[6],
                paths[7],
                paths[5],
                paths[0],
                paths[2],
                paths[3],
            ],
        )


class CheckpointManagerTest(unittest.TestCase):
    def test_create_checkpoint_manager(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = [
                f"{temp_dir}/epoch_1_step_3",
                f"{temp_dir}/epoch_0_step_1",
                f"{temp_dir}/epoch_0_step_5_loss=-0.3",
                f"{temp_dir}/epoch_1_step_1",
                f"{temp_dir}/epoch_1_step_2_loss=0.5",
                f"{temp_dir}/epoch_2_step_5_loss=0.3",
                f"{temp_dir}/epoch_0_step_2_acc=0.7",
                f"{temp_dir}/epoch_2_train_step_2_eval_step_5",
                f"{temp_dir}/epoch_3_train_step_5_loss=-0.2",
                f"{temp_dir}/epoch_3_eval_step_2_loss=0.2",
                f"{temp_dir}/epoch_3_train_step_1_eval_step_5_loss=0.1",
            ]
            for path in paths:
                os.mkdir(path)

            # without last_n_checkpoints
            ckpt_manager = CheckpointManager(temp_dir)
            self.assertEqual(ckpt_manager._ckpt_paths, [])

            # with last_n_checkpoints but without metric
            ckpt_manager = CheckpointManager(temp_dir, keep_last_n_checkpoints=2)
            self.assertEqual(
                [x.path for x in ckpt_manager._ckpt_paths],
                [
                    f"{temp_dir}/epoch_0_step_1",
                    f"{temp_dir}/epoch_0_step_2_acc=0.7",
                    f"{temp_dir}/epoch_0_step_5_loss=-0.3",
                    f"{temp_dir}/epoch_1_step_1",
                    f"{temp_dir}/epoch_1_step_2_loss=0.5",
                    f"{temp_dir}/epoch_1_step_3",
                    f"{temp_dir}/epoch_2_step_5_loss=0.3",
                    f"{temp_dir}/epoch_2_train_step_2_eval_step_5",
                    f"{temp_dir}/epoch_3_train_step_5_loss=-0.2",
                    f"{temp_dir}/epoch_3_eval_step_2_loss=0.2",
                    f"{temp_dir}/epoch_3_train_step_1_eval_step_5_loss=0.1",
                ],
            )

            # with last_n_checkpoints and metric min
            ckpt_manager = CheckpointManager(
                temp_dir,
                keep_last_n_checkpoints=3,
                best_checkpoint_config=BestCheckpointConfig(
                    monitored_metric="loss", mode="min"
                ),
            )
            self.assertEqual(
                [x.path for x in ckpt_manager._ckpt_paths],
                [
                    f"{temp_dir}/epoch_1_step_2_loss=0.5",
                    f"{temp_dir}/epoch_2_step_5_loss=0.3",
                    f"{temp_dir}/epoch_3_eval_step_2_loss=0.2",
                    f"{temp_dir}/epoch_3_train_step_1_eval_step_5_loss=0.1",
                    f"{temp_dir}/epoch_3_train_step_5_loss=-0.2",
                    f"{temp_dir}/epoch_0_step_5_loss=-0.3",
                ],
            )

            # with last_n_checkpoints and metric max
            ckpt_manager = CheckpointManager(
                temp_dir,
                keep_last_n_checkpoints=3,
                best_checkpoint_config=BestCheckpointConfig(
                    monitored_metric="loss", mode="max"
                ),
            )
            self.assertEqual(
                [x.path for x in ckpt_manager._ckpt_paths],
                [
                    f"{temp_dir}/epoch_0_step_5_loss=-0.3",
                    f"{temp_dir}/epoch_3_train_step_5_loss=-0.2",
                    f"{temp_dir}/epoch_3_train_step_1_eval_step_5_loss=0.1",
                    f"{temp_dir}/epoch_3_eval_step_2_loss=0.2",
                    f"{temp_dir}/epoch_2_step_5_loss=0.3",
                    f"{temp_dir}/epoch_1_step_2_loss=0.5",
                ],
            )

            # with last_n_checkpoints and non previously tracked metric
            ckpt_manager = CheckpointManager(
                temp_dir,
                keep_last_n_checkpoints=3,
                best_checkpoint_config=BestCheckpointConfig(
                    monitored_metric="foo", mode="max"
                ),
            )
            self.assertEqual(ckpt_manager._ckpt_paths, [])

        # More intense metric sorting test
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = [
                f"{temp_dir}/epoch_1_train_step_20_val_loss=0.09",
                f"{temp_dir}/epoch_0_train_step_10_eval_step_53412092_val_loss=10.0",
                f"{temp_dir}/epoch_4_step_130_val_loss=29.0",
                f"{temp_dir}/epoch_0_eval_step_2_val_loss=13.0",
                f"{temp_dir}/epoch_1_train_step_25_val_loss=0.06",
            ]
            for path in paths:
                os.mkdir(path)

            ckpt_manager = CheckpointManager(
                temp_dir,
                best_checkpoint_config=BestCheckpointConfig(
                    monitored_metric="val_loss", mode="min"
                ),
                keep_last_n_checkpoints=3,
            )
            self.assertEqual(
                [str(path) for path in ckpt_manager._ckpt_paths],
                [paths[2], paths[3], paths[1], paths[0], paths[4]],
            )

            ckpt_manager = CheckpointManager(
                temp_dir,
                best_checkpoint_config=BestCheckpointConfig(
                    monitored_metric="val_loss", mode="max"
                ),
                keep_last_n_checkpoints=3,
            )
            self.assertEqual(
                [str(path) for path in ckpt_manager._ckpt_paths],
                [paths[4], paths[0], paths[1], paths[3], paths[2]],
            )

    @skip_if_not_distributed
    def test_create_checkpoint_manager_distributed(self) -> None:
        spawn_multi_process(
            2,
            "gloo",
            self._test_create_checkpoint_manager_distributed,
        )

    @staticmethod
    def _test_create_checkpoint_manager_distributed() -> None:
        if get_global_rank() == 0:
            temp_dir = tempfile.mkdtemp()
            paths = ["epoch_1_step_2", "epoch_0_step_1", "epoch_1_step_1"]
            for path in paths:
                os.mkdir(os.path.join(temp_dir, path))
        else:
            temp_dir = ""

        tc = unittest.TestCase()

        # without top k config
        ckpt_manager = CheckpointManager(temp_dir)
        tc.assertNotEqual(ckpt_manager.dirpath, "")
        tc.assertEqual(ckpt_manager._ckpt_paths, [])

        # with top k config
        ckpt_manager = CheckpointManager(temp_dir, keep_last_n_checkpoints=1)
        tc.assertNotEqual(ckpt_manager.dirpath, "")
        tc.assertEqual(
            [str(x) for x in ckpt_manager._ckpt_paths],
            [
                os.path.join(ckpt_manager.dirpath, path)
                for path in [
                    "epoch_0_step_1",
                    "epoch_1_step_1",
                    "epoch_1_step_2",
                ]
            ],
        )

    def test_prune_surplus_checkpoints(self) -> None:
        # with checkpoints to delete
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_manager = CheckpointManager(temp_dir, keep_last_n_checkpoints=1)
            paths = [
                CheckpointPath(temp_dir, 0, 0),
                CheckpointPath(temp_dir, 0, 1),
                CheckpointPath(temp_dir, 1, 0),
            ]
            for path in paths:
                os.mkdir(path.path)

            ckpt_manager._ckpt_paths = list(paths)
            warning_messages = []
            expected_warning_msg = (
                f"3 checkpoints found in {temp_dir}. ",
                f"Deleting {2} oldest ",
                "checkpoints to enforce ``keep_last_n_checkpoints`` argument.",
            )
            with patch(
                f"{CheckpointManager.__module__}.logging.Logger.warning",
                warning_messages.append,
            ):
                ckpt_manager.prune_surplus_checkpoints()

            self.assertEqual(warning_messages[0], expected_warning_msg)
            self.assertEqual(ckpt_manager._ckpt_paths, [paths[2]])
            self.assertTrue(os.path.exists(paths[2].path))
            self.assertFalse(os.path.exists(paths[0].path))
            self.assertFalse(os.path.exists(paths[1].path))

        # without checkpoints to delete
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_manager = CheckpointManager(temp_dir)
            paths = [
                CheckpointPath(temp_dir, 0, 0),
                CheckpointPath(temp_dir, 0, 1),
                CheckpointPath(temp_dir, 1, 0),
            ]
            ckpt_manager._ckpt_paths = list(paths)
            ckpt_manager.prune_surplus_checkpoints()
            self.assertEqual(ckpt_manager._ckpt_paths, paths)

    def test_generate_checkpoint_path(self) -> None:
        ckpt_manager = CheckpointManager("foo")

        self.assertEqual(
            ckpt_manager.generate_checkpoint_path(1, 1).path,
            "foo/epoch_1_step_1",
        )

        self.assertEqual(
            ckpt_manager.generate_checkpoint_path(1, 3).path,
            "foo/epoch_1_step_3",
        )

        self.assertEqual(
            ckpt_manager.generate_checkpoint_path(
                1, {Phase.TRAIN: 5, Phase.EVALUATE: 7}
            ).path,
            "foo/epoch_1_train_step_5_eval_step_7",
        )

        ckpt_manager._best_checkpoint_config = BestCheckpointConfig(
            monitored_metric="val_loss", mode="min"
        )
        self.assertEqual(
            ckpt_manager.generate_checkpoint_path(
                1, 3, MetricData("val_loss", 0.5)
            ).path,
            "foo/epoch_1_step_3_val_loss=0.5",
        )

        # best checkpoint config, but did not pass metric data - expect path but no metric
        self.assertEqual(
            ckpt_manager.generate_checkpoint_path(1, 2).path,
            "foo/epoch_1_step_2",
        )

        # passed metric data is tracking a different metric than best checkpoint config - expect exception
        with self.assertRaisesRegex(
            AssertionError,
            "Attempted to get a checkpoint with metric 'mean', but best checkpoint config is for 'val_loss'",
        ):
            ckpt_manager.generate_checkpoint_path(1, 2, MetricData("mean", 3.5))

        # no best checkpoint config, but passed metric data - expect exception
        ckpt_manager._best_checkpoint_config = None
        with self.assertRaisesRegex(
            AssertionError,
            "Attempted to get a checkpoint with metric but best checkpoint config is not set",
        ):
            ckpt_manager.generate_checkpoint_path(1, 2, MetricData("val_loss", 3.5))

    @skip_if_not_distributed
    def test_generate_checkpoint_path_distributed(self) -> None:
        spawn_multi_process(
            world_size=2,
            backend="gloo",
            method=self._test_generate_checkpoint_path_distributed,
        )

    @staticmethod
    def _test_generate_checkpoint_path_distributed() -> None:
        tc = unittest.TestCase()

        init_from_env()

        ckpt_manager = CheckpointManager("foo")

        if dist.get_rank() == 0:
            path = ckpt_manager.generate_checkpoint_path(1, 1).path
        else:
            path = ckpt_manager.generate_checkpoint_path(3, 41).path

        tc.assertEqual(
            path,
            "foo/epoch_1_step_1",
        )

    def test_append_checkpoint_by_recency(self) -> None:
        ckpt_manager = CheckpointManager("foo", keep_last_n_checkpoints=3)
        ckpt_manager._ckpt_paths = [CheckpointPath("foo", 0, 0)]

        # without need to remove old by recency
        ckpt_manager.append_checkpoint(CheckpointPath("foo", 0, 1))
        self.assertEqual(
            ckpt_manager._ckpt_paths,
            [CheckpointPath("foo", 0, 0), CheckpointPath("foo", 0, 1)],
        )

        ckpt_manager.append_checkpoint(CheckpointPath("foo", 0, {Phase.TRAIN: 5}))
        self.assertEqual(
            ckpt_manager._ckpt_paths,
            [
                CheckpointPath("foo", 0, 0),
                CheckpointPath("foo", 0, 1),
                CheckpointPath("foo", 0, {Phase.TRAIN: 5}),
            ],
        )

        # removing old by recency
        with patch("fsspec.implementations.local.LocalFileSystem.rm") as mock_rm:
            ckpt_manager.append_checkpoint(
                CheckpointPath("foo", 0, {Phase.EVALUATE: 10})
            )
            self.assertEqual(
                ckpt_manager._ckpt_paths,
                [
                    CheckpointPath("foo", 0, 1),
                    CheckpointPath("foo", 0, {Phase.TRAIN: 5}),
                    CheckpointPath("foo", 0, {Phase.EVALUATE: 10}),
                ],
            )
            mock_rm.assert_called_once_with("foo/epoch_0_step_0", recursive=True)

    def test_append_checkpoint_by_metric(self) -> None:
        ckpt_manager = CheckpointManager(
            "foo",
            keep_last_n_checkpoints=5,
            best_checkpoint_config=BestCheckpointConfig(
                monitored_metric="val_loss", mode="max"
            ),
        )
        paths = [
            CheckpointPath(
                "foo", 0, x, metric_data=MetricData(name="val_loss", value=0.01 * x)
            )
            for x in range(1, 7, 1)
        ]
        ckpt_manager._ckpt_paths = [paths[1], paths[2], paths[4]]
        # without need to remove old by min metric, goes beginning
        ckpt_manager.append_checkpoint(paths[0])
        self.assertEqual(
            ckpt_manager._ckpt_paths,
            [paths[0], paths[1], paths[2], paths[4]],
        )
        # without need to remove old by min metric, goes end
        ckpt_manager.append_checkpoint(paths[5])
        self.assertEqual(
            ckpt_manager._ckpt_paths,
            [paths[0], paths[1], paths[2], paths[4], paths[5]],
        )
        # removing old max metric, goes middle
        with patch("fsspec.implementations.local.LocalFileSystem.rm") as mock_rm:
            ckpt_manager.append_checkpoint(paths[3])
            self.assertEqual(
                ckpt_manager._ckpt_paths,
                [paths[1], paths[2], paths[3], paths[4], paths[5]],
            )
            mock_rm.assert_called_once_with(
                "foo/epoch_0_step_1_val_loss=0.01", recursive=True
            )

        # no metric data - noop
        ckpt_manager._keep_last_n_checkpoints = None
        ckpt_manager.append_checkpoint(CheckpointPath("foo", 0, 8))
        self.assertEqual(
            ckpt_manager._ckpt_paths,
            [paths[1], paths[2], paths[3], paths[4], paths[5]],
        )

    def test_should_save_checkpoint(self) -> None:
        """
        Tests basic functionality of should_save_checkpoint
        """
        ckpt_manager = CheckpointManager("foo")

        # test default behavior
        ckpt = CheckpointPath("foo", 0, 2)
        self.assertTrue(ckpt_manager.should_save_checkpoint(ckpt))

        ckpt_manager._ckpt_paths = [CheckpointPath("foo", 0, 1)]
        self.assertTrue(ckpt_manager.should_save_checkpoint(ckpt))
        ckpt_manager._keep_last_n_checkpoints = 1
        self.assertTrue(ckpt_manager.should_save_checkpoint(ckpt))

        ckpt_manager._ckpt_paths = [
            CheckpointPath(
                "foo", 0, 1, metric_data=MetricData(name="val_loss", value=0.01)
            ),
        ]
        ckpt_manager._best_checkpoint_config = BestCheckpointConfig(
            monitored_metric="val_loss",
            mode="min",
        )

        bigger_metric = CheckpointPath(
            "foo", 0, 1, metric_data=MetricData(name="val_loss", value=0.02)
        )
        smaller_metric = CheckpointPath(
            "foo", 0, 1, metric_data=MetricData(name="val_loss", value=0.001)
        )
        ckpt_manager._keep_last_n_checkpoints = None
        self.assertTrue(ckpt_manager.should_save_checkpoint(bigger_metric))
        ckpt_manager._keep_last_n_checkpoints = 1
        self.assertFalse(ckpt_manager.should_save_checkpoint(bigger_metric))
        self.assertTrue(ckpt_manager.should_save_checkpoint(smaller_metric))
        ckpt_manager._keep_last_n_checkpoints = 2
        self.assertTrue(ckpt_manager.should_save_checkpoint(smaller_metric))
        self.assertTrue(ckpt_manager.should_save_checkpoint(bigger_metric))

        # Make sure we are actually comparing against more optimal element
        ckpt_manager._ckpt_paths = [
            CheckpointPath(
                "foo", 0, 1, metric_data=MetricData(name="val_loss", value=0.01)
            ),
            CheckpointPath(
                "foo", 0, 1, metric_data=MetricData(name="val_loss", value=0.05)
            ),
        ]

        ckpt_manager._best_checkpoint_config = BestCheckpointConfig(
            monitored_metric="val_loss",
            mode="max",
        )
        ckpt_manager._keep_last_n_checkpoints = 2
        self.assertTrue(ckpt_manager.should_save_checkpoint(bigger_metric))

    def test_remove_worst_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.mkdir(os.path.join(temp_dir, "epoch_0_step_0"))
            os.mkdir(os.path.join(temp_dir, "epoch_0_step_1"))

            ckpt_manager = CheckpointManager(temp_dir)
            ckpt_manager.append_checkpoint(CheckpointPath(temp_dir, 0, 0))
            ckpt_manager.append_checkpoint(CheckpointPath(temp_dir, 0, 1))

            ckpt_manager.remove_checkpoint()
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "epoch_0_step_0")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "epoch_0_step_1")))
            self.assertEqual(ckpt_manager._ckpt_paths, [CheckpointPath(temp_dir, 0, 1)])

    @patch(
        "fsspec.implementations.local.LocalFileSystem.rm",
        side_effect=Exception("OSError: [Errno 2] No such file or directory"),
    )
    def test_remove_worst_checkpoint_exception(self, mock_url_to_fs: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.mkdir(os.path.join(temp_dir, "epoch_0_train_step_0"))
            os.mkdir(os.path.join(temp_dir, "epoch_0_train_step_1"))

            ckpt_manager = CheckpointManager(temp_dir, keep_last_n_checkpoints=2)

            log_container = []
            with patch(
                "torchtnt.utils.checkpoint.logging.Logger.error", log_container.append
            ):
                ckpt_manager.append_checkpoint(
                    CheckpointPath(temp_dir, 0, {Phase.TRAIN: 2})
                )

            self.assertEqual(
                log_container,
                [
                    (
                        f"Failed to remove checkpoint '{temp_dir}/epoch_0_train_step_0' for bookkeeping purposes. "
                        "Do not use it to restore since it may be corrupted! Exception: OSError: [Errno 2] No such file or directory"
                    )
                ],
            )
            # Make sure we are not tracking the oldest one anymore, even if it was not deleted
            self.assertEqual(
                ckpt_manager._ckpt_paths,
                [
                    CheckpointPath(temp_dir, 0, {Phase.TRAIN: 1}),
                    CheckpointPath(temp_dir, 0, {Phase.TRAIN: 2}),
                ],
            )


class CheckpointUtilsTest(unittest.TestCase):
    @staticmethod
    def _create_snapshot_metadata(output_dir: str) -> None:
        path = os.path.join(output_dir, METADATA_FNAME)
        with open(path, "w"):
            pass

    def test_latest_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(get_latest_checkpoint_path(temp_dir))

        with tempfile.TemporaryDirectory() as temp_dir:
            latest_path = os.path.join(temp_dir, "epoch_0_step_0")
            os.mkdir(latest_path)
            self.assertEqual(
                get_latest_checkpoint_path(temp_dir),
                latest_path,
            )
            self.assertEqual(
                get_latest_checkpoint_path(temp_dir, METADATA_FNAME),
                None,
            )
            self._create_snapshot_metadata(latest_path)
            self.assertEqual(
                get_latest_checkpoint_path(temp_dir, METADATA_FNAME),
                latest_path,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            path_1 = os.path.join(temp_dir, "epoch_0_step_0")
            os.mkdir(path_1)
            self._create_snapshot_metadata(path_1)
            path_2 = os.path.join(temp_dir, "epoch_0_step_100_val_loss=0.002")
            os.mkdir(path_2)
            self._create_snapshot_metadata(path_2)

            # Missing metadata file
            path_3 = os.path.join(temp_dir, "epoch_1_step_100")
            os.mkdir(path_3)

            # Ill-formatted name
            path_4 = os.path.join(temp_dir, "epoch_700")
            os.mkdir(path_4)
            self.assertEqual(
                get_latest_checkpoint_path(temp_dir, METADATA_FNAME),
                path_2,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            path_1 = os.path.join(temp_dir, "epoch_0_train_step_0")
            os.mkdir(path_1)
            self._create_snapshot_metadata(path_1)
            path_2 = os.path.join(temp_dir, "epoch_0_train_step_100_val_loss=0.002")
            os.mkdir(path_2)
            self._create_snapshot_metadata(path_2)

            self.assertEqual(
                get_latest_checkpoint_path(temp_dir, METADATA_FNAME), path_2
            )

            path_3 = os.path.join(temp_dir, "epoch_0_eval_step_0")
            os.mkdir(path_3)
            self._create_snapshot_metadata(path_3)
            self.assertEqual(
                get_latest_checkpoint_path(temp_dir, METADATA_FNAME), path_3
            )

            # Missing metadata file
            path_4 = os.path.join(temp_dir, "epoch_1_train_step_1")
            os.mkdir(path_4)
            self.assertEqual(get_latest_checkpoint_path(temp_dir), path_4)
            self.assertEqual(
                get_latest_checkpoint_path(temp_dir, METADATA_FNAME), path_3
            )
            self._create_snapshot_metadata(path_4)

            # Ill-formatted name
            path_5 = os.path.join(temp_dir, "epoch_700")
            os.mkdir(path_5)
            self.assertEqual(
                get_latest_checkpoint_path(temp_dir, METADATA_FNAME), path_4
            )

            # With both phases
            path_6 = os.path.join(temp_dir, "epoch_1_train_step_5_eval_step_5")
            os.mkdir(path_6)
            self._create_snapshot_metadata(path_6)
            self.assertEqual(
                get_latest_checkpoint_path(temp_dir, METADATA_FNAME), path_6
            )

            path_9 = os.path.join(temp_dir, "epoch_1_train_step_5_eval_step_10")
            os.mkdir(path_9)
            self._create_snapshot_metadata(path_9)
            self.assertEqual(
                get_latest_checkpoint_path(temp_dir, METADATA_FNAME), path_9
            )

            path_10 = os.path.join(temp_dir, "epoch_2_train_step_10_eval_step_10")
            os.mkdir(path_10)
            self._create_snapshot_metadata(path_10)
            self.assertEqual(
                get_latest_checkpoint_path(temp_dir, METADATA_FNAME), path_10
            )

            path_11 = os.path.join(temp_dir, "epoch_2_train_step_15_eval_step_10")
            os.mkdir(path_11)
            self._create_snapshot_metadata(path_11)
            self.assertEqual(
                get_latest_checkpoint_path(temp_dir, METADATA_FNAME), path_11
            )

            # Test legacy path with fewer steps

            # Test legacy path with equal steps

            # Test legacy path with more steps

    @skip_if_not_distributed
    def test_latest_checkpoint_path_distributed(self) -> None:
        spawn_multi_process(
            2,
            "gloo",
            self._latest_checkpoint_path_distributed,
        )

    @staticmethod
    def _latest_checkpoint_path_distributed() -> None:
        tc = unittest.TestCase()
        is_rank0 = get_global_rank() == 0

        if is_rank0:
            temp_dir = tempfile.mkdtemp()
        else:
            temp_dir = ""
        tc.assertIsNone(get_latest_checkpoint_path(temp_dir))
        if is_rank0:
            shutil.rmtree(temp_dir)  # delete temp directory

        if is_rank0:
            temp_dir = tempfile.mkdtemp()
            path_1 = os.path.join(temp_dir, "epoch_0_step_0")
            os.mkdir(path_1)
            CheckpointUtilsTest._create_snapshot_metadata(path_1)
            path_2 = os.path.join(temp_dir, "epoch_0_step_100")
            os.mkdir(path_2)
            CheckpointUtilsTest._create_snapshot_metadata(path_2)

            # Missing metadata file
            path_3 = os.path.join(temp_dir, "epoch_1_step_100")
            os.mkdir(path_3)

            # Ill-formatted name
            path_4 = os.path.join(temp_dir, "epoch_700")
            os.mkdir(path_4)
        else:
            temp_dir = ""
            path_2 = ""

        pg = PGWrapper(dist.group.WORLD)
        path_container = [path_2] if is_rank0 else [None]
        pg.broadcast_object_list(path_container, 0)
        expected_path = path_container[0]
        tc.assertIsNotNone(expected_path)
        tc.assertEqual(
            get_latest_checkpoint_path(temp_dir, METADATA_FNAME),
            expected_path,
        )

        if is_rank0:
            shutil.rmtree(temp_dir)  # delete temp directory

    def test_best_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(get_best_checkpoint_path(temp_dir, "val_loss", "min"))

            # no checkpoint w/ metric value
            path = os.path.join(temp_dir, "epoch_0_step_0")
            os.mkdir(path)
            self.assertIsNone(get_best_checkpoint_path(temp_dir, "val_loss", "min"))

        with tempfile.TemporaryDirectory() as temp_dir:
            best_path = os.path.join(temp_dir, "epoch_0_step_0_val_loss=0.01")
            os.mkdir(best_path)
            self.assertEqual(
                get_best_checkpoint_path(temp_dir, "val_loss", "min"),
                best_path,
            )
            self.assertIsNone(
                get_best_checkpoint_path(temp_dir, "val_loss", "min", METADATA_FNAME),
                None,
            )
            self._create_snapshot_metadata(best_path)
            self.assertEqual(
                get_best_checkpoint_path(temp_dir, "val_loss", "min", METADATA_FNAME),
                best_path,
            )

            # handle negative values
            best_path_2 = os.path.join(temp_dir, "epoch_0_step_0_val_loss=-0.01")
            os.mkdir(best_path_2)
            self.assertEqual(
                get_best_checkpoint_path(temp_dir, "val_loss", "min"),
                best_path_2,
            )

            # handle "max" mode correctly
            best_path_3 = os.path.join(
                temp_dir, "epoch_0_train_step_100_eval_step_25_val_loss=0.1"
            )
            os.mkdir(best_path_3)
            self.assertEqual(
                get_best_checkpoint_path(temp_dir, metric_name="val_loss", mode="max"),
                best_path_3,
            )

            # handle different metric correctly
            best_path_4 = os.path.join(temp_dir, "epoch_0_step_100_train_loss=0.2")
            os.mkdir(best_path_4)
            self.assertEqual(
                get_best_checkpoint_path(temp_dir, metric_name="val_loss", mode="max"),
                best_path_3,
            )
            self.assertEqual(
                get_best_checkpoint_path(
                    temp_dir, metric_name="train_loss", mode="max"
                ),
                best_path_4,
            )

    def test_retrieve_checkpoint_dirpaths(self) -> None:
        """
        Tests retrieving checkpoint directories from a given root directory
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = [
                "epoch_0_step_10",
                "epoch_2_step_10",
                "epoch_0_eval_step_5",
                "epoch_0_step_6",
                "epoch_0_train_step_3",
                "epoch_0_step_10_train_loss=13.0",
                "epoch_34_eval_step_10_val_loss=10.0",
                "epoch_3_train_step_10_val_loss=5.1",
                "epoch_1_step_10",
                "epoch_25_train_step_10_eval_step_15",
                "epoch_1_train_step_10_eval_step_0_train_loss=10.0",
            ]
            for path in paths[:-1]:
                os.mkdir(os.path.join(temp_dir, path))
            # make last path a file instead of a directory
            with open(os.path.join(temp_dir, paths[-1]), "w"):
                pass

            # compares set equality since order of returned dirpaths is not guaranteed
            # in _retrieve_checkpoint_dirpaths
            self.assertEqual(
                {
                    str(x)
                    for x in _retrieve_checkpoint_dirpaths(
                        temp_dir, metadata_fname=None
                    )
                },
                {os.path.join(temp_dir, path) for path in paths[:-1]},
            )
            self.assertEqual(
                _retrieve_checkpoint_dirpaths(temp_dir, metadata_fname=".metadata"),
                [],
            )

            # check metadata file is correct filtered for
            # by creating metadata for 3rd and 6th path in list
            with open(os.path.join(temp_dir, paths[2], ".metadata"), "w"):
                pass

            with open(os.path.join(temp_dir, paths[5], ".metadata"), "w"):
                pass

            self.assertEqual(
                {
                    str(x)
                    for x in _retrieve_checkpoint_dirpaths(
                        temp_dir, metadata_fname=".metadata"
                    )
                },
                {os.path.join(temp_dir, paths[2]), os.path.join(temp_dir, paths[5])},
            )

    def test_retrieve_checkpoint_dirpaths_with_metrics(self) -> None:
        """
        Tests retrieving checkpoint (w/ metrics) directories from a given root directory
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = [
                "epoch_0_train_step_10_val_loss=10.0",
                "epoch_1_step_10_val_loss=5.0",
                "epoch_0_train_step_7_eval_step_12_val_loss=12.9",
                "epoch_2_eval_step_10",
                "epoch_0_train_step_5",
                "epoch_0_train_step_6_train_loss=13.0",
            ]
            for path in paths:
                os.mkdir(os.path.join(temp_dir, path))
            # make last path a file instead of a directory
            with open(os.path.join(temp_dir, "epoch_0_step_3_val_loss=3.0"), "w"):
                pass

            # compares set equality since order of returned dirpaths is not guaranteed
            # in _retrieve_checkpoint_dirpaths
            self.assertEqual(
                {
                    str(x)
                    for x in _retrieve_checkpoint_dirpaths(
                        temp_dir, metadata_fname=None
                    )
                },
                {os.path.join(temp_dir, path) for path in paths},
            )
            self.assertEqual(
                {
                    str(x)
                    for x in _retrieve_checkpoint_dirpaths(
                        temp_dir, metadata_fname=None, metric_name="val_loss"
                    )
                },
                {
                    os.path.join(temp_dir, path) for path in paths[:3]
                },  # since last path is a file
            )
            self.assertEqual(
                _retrieve_checkpoint_dirpaths(temp_dir, metadata_fname=".metadata"),
                [],
            )

            # check metadata file is correct filtered for
            # by creating metadata for 3rd path in list
            with open(os.path.join(temp_dir, paths[1], ".metadata"), "w"):
                pass

            self.assertEqual(
                {
                    str(x)
                    for x in _retrieve_checkpoint_dirpaths(
                        temp_dir, metadata_fname=".metadata", metric_name="val_loss"
                    )
                },
                {os.path.join(temp_dir, paths[1])},
            )

    @skip_if_not_distributed
    def test_distributed_get_checkpoint_dirpaths(self) -> None:
        spawn_multi_process(2, "gloo", self._distributed_get_checkpoint_dirpaths)

    @staticmethod
    def _distributed_get_checkpoint_dirpaths() -> None:
        """
        Tests that existing checkpoint directories are read and
        properly registered on all ranks
        """

        @rank_zero_read_and_broadcast
        def create_tmp_dir() -> str:
            return tempfile.mkdtemp()

        init_from_env()

        temp_dir = create_tmp_dir()
        try:
            path1 = os.path.join(temp_dir, "epoch_0_step_10")
            path2 = os.path.join(temp_dir, "epoch_1_step_20")
            if get_global_rank() == 0:
                os.mkdir(path1)
                os.mkdir(path2)
            torch.distributed.barrier()

            ckpt_dirpaths = [str(x) for x in get_checkpoint_dirpaths(temp_dir)]
            tc = unittest.TestCase()
            tc.assertEqual(set(ckpt_dirpaths), {path1, path2})

            tc.assertEqual(
                get_checkpoint_dirpaths(temp_dir, metadata_fname=".metadata"), []
            )
        finally:
            dist.barrier()  # avoid race condition
            if get_global_rank() == 0:
                shutil.rmtree(temp_dir)  # delete temp directory

    def test_get_checkpoint_dirpaths(self) -> None:
        """
        Tests that `get_checkpoint_dirpaths` returns
        the sorted checkpoint directories correctly
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path1 = os.path.join(temp_dir, "epoch_1_step_20")
            path2 = os.path.join(temp_dir, "epoch_4_eval_step_130")
            path3 = os.path.join(temp_dir, "epoch_0_step_10")
            path4 = os.path.join(
                temp_dir, "epoch_0_train_step_10_eval_step_15_train_loss=13.0"
            )
            malformed_path = os.path.join(
                temp_dir, "epoch_train_0_step_10_val_loss=10.0"
            )
            os.mkdir(path1)
            os.mkdir(path2)
            os.mkdir(path3)
            os.mkdir(path4)
            os.mkdir(malformed_path)

            self.assertEqual(
                {str(x) for x in get_checkpoint_dirpaths(temp_dir)},
                {path1, path2, path3, path4},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            path1 = os.path.join(temp_dir, "epoch_1_step_20_val_loss=0.01")
            path2 = os.path.join(temp_dir, "epoch_4_train_step_130_val_loss=-0.2")
            path3 = os.path.join(temp_dir, "epoch_0_eval_step_10_val_loss=0.12")
            path4 = os.path.join(
                temp_dir, "epoch_0_train_step_10_eval_step_15_val_loss=13.0"
            )
            os.mkdir(path1)
            os.mkdir(path2)
            os.mkdir(path3)

            self.assertEqual(
                {
                    str(x)
                    for x in get_checkpoint_dirpaths(temp_dir, metric_name="val_loss")
                },
                {path1, path2, path3},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(
                get_checkpoint_dirpaths(temp_dir),
                [],
            )

    def test_get_checkpoint_dirpaths_with_multiple_metadata_fnames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path1 = os.path.join(temp_dir, "epoch_1_step_20")
            os.mkdir(path1)

            path2 = os.path.join(temp_dir, "epoch_4_eval_step_130")
            os.mkdir(path2)

            with open(os.path.join(path1, ".metadata"), "w"):
                pass

            with open(os.path.join(path2, ".manifest"), "w"):
                pass

            self.assertEqual(
                [
                    str(x)
                    for x in get_checkpoint_dirpaths(
                        temp_dir, metadata_fname=[".metadata"]
                    )
                ],
                [path1],
            )

            self.assertEqual(
                {
                    str(x)
                    for x in get_checkpoint_dirpaths(
                        temp_dir, metadata_fname=[".manifest", ".metadata"]
                    )
                },
                {path1, path2},
            )

    def test_metadata_exists(self) -> None:
        app_state = {"module": nn.Linear(2, 2)}
        with tempfile.TemporaryDirectory() as temp_dir:
            dirpath = os.path.join(temp_dir, "checkpoint")
            Snapshot.take(dirpath, app_state=app_state)

            fs = get_filesystem(dirpath)
            self.assertTrue(_metadata_exists(fs, dirpath, SNAPSHOT_METADATA_FNAME))

            os.remove(os.path.join(dirpath, SNAPSHOT_METADATA_FNAME))
            self.assertFalse(_metadata_exists(fs, dirpath, SNAPSHOT_METADATA_FNAME))

    def test_does_checkpoint_metadata_exist(self) -> None:
        app_state = {"module": nn.Linear(2, 2)}

        with tempfile.TemporaryDirectory() as temp_dir:
            dirpath = os.path.join(temp_dir, "checkpoint")
            Snapshot.take(dirpath, app_state=app_state)

            self.assertTrue(does_checkpoint_exist(dirpath, SNAPSHOT_METADATA_FNAME))

            os.remove(os.path.join(dirpath, SNAPSHOT_METADATA_FNAME))
            self.assertFalse(does_checkpoint_exist(dirpath, SNAPSHOT_METADATA_FNAME))

    def test_does_checkpoint_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_1 = os.path.join(temp_dir, "checkpoint_1")
            os.mkdir(ckpt_1)

            # pyre-fixme[6]: For 2nd argument expected `Union[List[str], str]` but
            #  got `None`.
            self.assertFalse(does_checkpoint_exist(ckpt_1, metadata_fname=None))

            with open(os.path.join(ckpt_1, ".metadata"), "w"):
                pass

            self.assertFalse(does_checkpoint_exist(ckpt_1, metadata_fname="manifest"))
            self.assertTrue(does_checkpoint_exist(ckpt_1, metadata_fname=".metadata"))
            self.assertTrue(
                does_checkpoint_exist(ckpt_1, metadata_fname=["manifest", ".metadata"])
            )
            self.assertFalse(
                does_checkpoint_exist(
                    ckpt_1, metadata_fname=["manifest", ".state_dict_info"]
                )
            )


class MyValLossUnit(TrainUnit[Batch]):
    def __init__(self) -> None:
        super().__init__()
        self.val_loss = 0.01

    def train_step(self, state: State, data: Batch) -> None:
        return None
