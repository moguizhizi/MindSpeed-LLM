import os
import shutil
import sys
from pathlib import Path
import pytest
import torch.distributed as dist

import mindspeed_llm
from tests.test_tools.utils import build_args, create_testconfig, run_cmd, weight_compare


def init_process_group(backend='nccl'):
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if not dist.is_initialized():
        dist.init_process_group(backend)


init_process_group()


class TestCovertPhi35CkptHf2mg:
    world_size = 8
    cur_dir = Path(__file__).parent
    json_file = next(cur_dir.glob("*.json"), None)
    test_config = create_testconfig(json_file)

    @pytest.mark.parametrize("params", test_config["test_phi35_moe_hf2mg_tp1pp8"])
    def test_phi35_moe_hf2mg_tp1pp8(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        BASE_DIR = Path(__file__).absolute().parents[3]
        CKPT_PYPATH = os.path.join(BASE_DIR, "convert_ckpt.py")
        exit_code = run_cmd(["python", CKPT_PYPATH] + sys.argv[1:])
        assert exit_code == 0
        base_dir = '/data/pipe/phi35-moe-tp1pp8-mcore-base'
        save_dir = '/data/pipe/phi35-moe-tp1pp8-mcore-test'
        assert weight_compare(base_dir, save_dir)
        shutil.rmtree(save_dir)
