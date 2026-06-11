# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import shutil
import tempfile
import unittest

import torch.distributed as dist
from torchtnt.utils import init_from_env
from torchtnt.utils.checkpoint import get_checkpoint_dirpaths
from torchtnt.utils.distributed import get_global_rank, spawn_multi_process
from torchtnt.utils.test_utils import skip_if_not_distributed, skip_if_not_gpu


class TestCheckpointUtilsGPU(unittest.TestCase):

    @skip_if_not_distributed
    @skip_if_not_gpu
    def test_get_checkpoint_dirpaths_distributed(self) -> None:
        spawn_multi_process(
            2, "nccl", self._test_get_checkpoint_dirpaths, timeout_s=180
        )

    @staticmethod
    def _test_get_checkpoint_dirpaths() -> None:
        """
        Tests retrieving checkpoint directories from a given root directory
        using NCCL on GPUs with custom state for pickling.
        """
        init_from_env()
        paths = [
            "epoch_0_step_10",
            "epoch_1_step_10_val_loss=10.5",
            "epoch_2_step_10",
            "epoch_0_step_5",
            "epoch_0_step_6_acc=0.03",
            "epoch_0_step_3",
        ]

        if get_global_rank() == 0:
            temp_dir = tempfile.mkdtemp()
            for path in paths:
                os.mkdir(os.path.join(temp_dir, path))
        else:
            temp_dir = None

        tc = unittest.TestCase()
        # Only rank 0 will know about temp_dir
        if get_global_rank() != 0:
            tc.assertIsNone(temp_dir)

        ckpt_dirpaths = get_checkpoint_dirpaths(
            # pyre-fixme[6]: For 1st argument expected `str` but got `Optional[str]`.
            temp_dir,
            process_group=dist.group.WORLD,
        )

        # Broadcast temp_dir to verify successful execution
        temp_dir = [temp_dir] if get_global_rank() == 0 else [None]
        dist.broadcast_object_list(temp_dir, src=0, group=dist.group.WORLD)
        temp_dir = temp_dir[0]
        tc.assertIsNotNone(temp_dir)

        tc.assertEqual(
            {str(x) for x in ckpt_dirpaths},
            # pyrefly: ignore [no-matching-overload]
            {os.path.join(temp_dir, path) for path in paths},
        )

        if get_global_rank() == 0:
            # pyrefly: ignore [bad-argument-type]
            shutil.rmtree(temp_dir)
