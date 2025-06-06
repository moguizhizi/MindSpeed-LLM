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

import types
from functools import wraps
from typing import Union

from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.training import get_args
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

from mindspeed_llm.core.transformer.transformer_layer import TransformerLayer
from mindspeed_llm.core.transformer.custom_layers.transformer_engine import PTNorm
from mindspeed_llm.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlockSubmodules,
    get_mtp_layer_offset,
    get_mtp_layer_spec,
    get_mtp_num_layers_to_build,
)


def get_gpt_layer_local_spec_wrapper(fn):
    @wraps(fn)
    def wrapper(num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False):
        res = fn(num_experts, moe_grouped_gemm, qk_layernorm)

        res.submodules.input_layernorm = PTNorm
        res.submodules.pre_mlp_layernorm = PTNorm

        if qk_layernorm:
            res.submodules.self_attention.submodules.q_layernorm = PTNorm
            res.submodules.self_attention.submodules.k_layernorm = PTNorm
        return res

    return wrapper


def build_layers_wrapper(fn, column_forward, row_forward):
    """
    For MOE + Ascend MC2, we replace linear_fc1 and linear_fc2 with vanilla column_forward and row_forward in megatron.
    """

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        if not get_args().use_mc2:
            return
        for layer in self.layers:
            if isinstance(getattr(layer, 'mlp', None), MoELayer):
                for local_expert in layer.mlp.experts.local_experts:
                    local_expert.linear_fc1.forward = types.MethodType(column_forward, local_expert.linear_fc1)
                    local_expert.linear_fc2.forward = types.MethodType(row_forward, local_expert.linear_fc2)

    return wrapper


def get_gpt_mtp_block_spec(
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        use_transformer_engine: bool,
) -> MultiTokenPredictionBlockSubmodules:
    """GPT Multi-Token Prediction (MTP) block spec."""
    num_layers_to_build = get_mtp_num_layers_to_build(config)
    if num_layers_to_build == 0:
        return None

    if isinstance(spec, TransformerBlockSubmodules):
        # get the spec for the last layer of decoder block
        transformer_layer_spec = spec.layer_specs[-1]
    elif isinstance(spec, ModuleSpec) and spec.module == TransformerLayer:
        transformer_layer_spec = spec
    else:
        raise ValueError(f"Invalid spec: {spec}")

    mtp_layer_spec = get_mtp_layer_spec(
        transformer_layer_spec=transformer_layer_spec, use_transformer_engine=use_transformer_engine
    )
    mtp_num_layers = config.mtp_num_layers if config.mtp_num_layers else 0
    mtp_layer_specs = [mtp_layer_spec] * mtp_num_layers

    offset = get_mtp_layer_offset(config)
    # split the mtp layer specs to only include the layers that are built in this pipeline stage.
    mtp_layer_specs = mtp_layer_specs[offset: offset + num_layers_to_build]
    if len(mtp_layer_specs) > 0:
        if len(mtp_layer_specs) != config.mtp_num_layers:
            raise AssertionError(f"currently all of the mtp layers must stage in the same pipeline stage.")
        mtp_block_spec = MultiTokenPredictionBlockSubmodules(layer_specs=mtp_layer_specs)
    else:
        mtp_block_spec = None

    return mtp_block_spec
