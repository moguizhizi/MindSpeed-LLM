# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from mindspeed_llm.tasks.trainer.base import BaseTrainer
from mindspeed_llm.tasks.rl.dpo import DPOTrainer
from mindspeed_llm.tasks.rl.rm import RMTrainer
from mindspeed_llm.tasks.rl.simpo import SimPOTrainer


class AutoTrainer:
    """
    A trainer is used to automatically determine which scenario (common pt, sft, rm, dpo) to use.
    """

    def __init__(self):
        initialize_megatron()

        self.args = get_args()

        if self.args.stage == "dpo":
            self.trainer = DPOTrainer()
        elif self.args.stage == "rm":
            self.trainer = RMTrainer()
        elif self.args.stage == "simpo":
            self.trainer = SimPOTrainer()
        else:
            self.trainer = BaseTrainer()

    def train(self):
        self.trainer.train()
