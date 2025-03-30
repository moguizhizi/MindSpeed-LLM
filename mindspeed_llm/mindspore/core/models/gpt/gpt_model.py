# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from functools import wraps
from typing import List

import torch
from torch import Tensor

from megatron.core import InferenceParams, tensor_parallel, parallel_state
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import build_module
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.training import get_args
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.transformer import ModuleSpec
from mindspeed_llm.core.transformer.custom_layers.transformer_engine import PTNorm

from mindspeed_llm.core.tensor_parallel.layers import SegmentedColumnParallelLinear
from mindspeed_llm.mindspore.tasks.models.transformer.multi_token_predication import MultiTokenPredication, MultiTokenPredicationSubmodules
from mindspeed_llm.core.models.gpt.gpt_model import setup_mtp_embeddings_layer

# Use this spec for multi token predication
mtp_sepc = ModuleSpec(
    module=MultiTokenPredication,
    submodules=MultiTokenPredicationSubmodules(
        embedding=None,
        enorm=PTNorm,
        hnorm=PTNorm,
        eh_proj=ColumnParallelLinear,
        transformer_layer=None,
        final_layernorm=PTNorm,
        output_layer=None,
    )
)



def gpt_model_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        post_layer_norm = kwargs.pop('post_layer_norm', True)
        fn(self, *args, **kwargs)
        config = args[1] if len(args) > 1 else kwargs['config']
        arguments = get_args()
        if self.post_process and arguments.add_output_layer_bias:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=True,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

        if self.post_process and arguments.output_layer_slice_num > 1:
            self.output_layer = SegmentedColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )
        if not post_layer_norm:
            self.decoder.post_layer_norm = False
        self.num_nextn_predict_layers = arguments.num_nextn_predict_layers
        self.share_mtp_embedding_and_output_weight = arguments.share_mtp_embedding_and_output_weight
        if self.post_process and self.training and self.num_nextn_predict_layers:
            self.mtp_layers = torch.nn.ModuleList(
                [
                    MultiTokenPredication(
                        config,
                        self.transformer_layer_spec,
                        mtp_sepc.submodules,
                        vocab_size=self.vocab_size,
                        max_sequence_length=self.max_sequence_length,
                        layer_number=i,
                        pre_process=self.pre_process,
                        post_process=self.post_process,
                        fp16_lm_cross_entropy=kwargs.get("fp16_lm_cross_entropy", False),
                        parallel_output=self.parallel_output,
                        position_embedding_type=self.position_embedding_type,
                        rotary_percent=kwargs.get("rotary_percent", 1.0),
                        seq_len_interpolation_factor=kwargs.get("rotary_seq_len_interpolation_factor", None),
                        share_mtp_embedding_and_output_weight=self.share_mtp_embedding_and_output_weight,
                    )
                    for i in range(self.num_nextn_predict_layers)
                ]
            )

        if self.post_process and self.num_nextn_predict_layers:
            # move block main model final norms here
            self.final_layernorm = build_module(
                    TENorm,
                    config=self.config,
                    hidden_size=self.config.hidden_size,
                    eps=self.config.layernorm_epsilon,
                )
        else:
            self.final_layernorm = None

        if self.pre_process or self.post_process:
            setup_mtp_embeddings_layer(self)

    return wrapper