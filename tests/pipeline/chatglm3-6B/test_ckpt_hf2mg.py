import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import mindspeed_llm
from convert_ckpt import main
from tests.test_tools.dist_test import DistributedTest
from tests.test_tools.utils import (build_args, create_testconfig, run_cmd,
                                    weight_compare)

PATTERN = r"acc = (.*)"


def init_process_group(backend='nccl'):
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if not dist.is_initialized():
        dist.init_process_group(backend)

init_process_group()


class TestConvertChatglm3CkptHf2mg():
    world_size = 8
    cur_dir = Path(__file__).parent
    json_file = next(cur_dir.glob("*.json"), None)
    test_config = create_testconfig(json_file)

    @pytest.mark.parametrize("params", test_config["test_chatglm3_hf2legacy_tp2pp4"])
    def test_chatglm3_hf2legacy_tp2pp4(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        BASE_DIR = Path(__file__).absolute().parents[3]
        CKPT_PYPATH = os.path.join(BASE_DIR, "convert_ckpt.py")
        arguments = [f"--{k}={v}" for k, v in params.items()]
        arguments.append("--add-qkv-bias")
        exit_code = run_cmd(["python3", CKPT_PYPATH] + arguments)
        assert exit_code == 0
        base_dir = '/data/chatglm3-6b-base-mg-tp2pp4-legacy-base/'
        save_dir = '/data/chatglm3-6b-base-mg-tp2pp4-legacy-test/'
        assert weight_compare(base_dir, save_dir)
        shutil.rmtree(save_dir)