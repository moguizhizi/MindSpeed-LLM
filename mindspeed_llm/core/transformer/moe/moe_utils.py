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

from functools import wraps

import torch
from megatron.core import parallel_state
from megatron.core.transformer.moe.moe_utils import get_capacity
from megatron.training import get_args


def z_loss_func(logits, z_loss_coeff):
    """Encourages the router's logits to remain small to enhance stability.
    Please refer to the ST-MoE paper for details.
    adapter for logsumexp() to support bfloat16

    Args:
        logits (torch.Tensor): The logits of the router.

    Returns:
        torch.Tensor: The logits after applying the z-loss.
    """

    z_loss = torch.mean(torch.square(torch.logsumexp(logits.to(torch.float), dim=-1).to(logits.dtype))) * z_loss_coeff
    return z_loss


def device_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_tokens: int,
    num_experts: int,
    moe_router_topk_limited_devices: int,
):
    """Perform top-k routing on a subset of expert parallel ranks.

    Selects N ranks for each token, then conducts top-k selection among experts on these devices.
    See DeepSeek-V2 technical report (https://arxiv.org/pdf/2405.04434) for details.

    Args:
        scores (torch.Tensor): Softmax scores from the router.
        topk (int): The number of experts to select for each token.
        num_tokens (int): The number of tokens.
        num_experts (int): The number of experts.
        moe_router_topk_limited_devices (int): Number of expert parallel ranks to consider for
            each token during routing. None means no device limitation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Probs and indices tensor.
    """

    # Organize the experts into groups
    num_group = (
        parallel_state.get_expert_model_parallel_world_size()
    )  # num_group equals to expert parallel size
    group_scores = scores.view(num_tokens, num_group, -1).max(dim=-1).values
    group_idx = torch.topk(group_scores, k=moe_router_topk_limited_devices, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    # Mask the experts based on selection groups
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_group, num_experts // num_group)
        .reshape(num_tokens, -1)
    )

    masked_scores = scores.masked_fill(~score_mask.bool(), 0.0)
    probs, top_indices = torch.topk(masked_scores, k=topk, dim=-1)

    return probs, top_indices


def topk_softmax_with_capacity(
    logits: torch.Tensor,
    topk: int,
    capacity_factor: float = None,
    pad_to_capacity: bool = False,
    drop_policy: str = "probs",
    use_pre_softmax: bool = False,
    moe_router_topk_limited_devices: int = None,
    moe_router_topk_scaling_factor: float = None,
    deterministic_mode: bool = False,
    score_function: str = "softmax",
    expert_bias: torch.Tensor = None,
    norm_topk_prob=False,
):
    """Apply capacity and padding to the top-k selection.
    Args:
        logits (torch.Tensor): Logits tensor.
        topk (int): The number of experts to select for each token.
        capacity_factor (int): The capacity factor of each expert. Will drop tokens if the number of tokens exceeds the capacity.
        pad_to_capacity (bool): Whether to need padding in token drop mode.
        drop_policy (str): The policy to drop tokens. Can be either "prob" or "position". If "prob", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Probs, indices and tokens_per_expert tensor.

        (1) If there's no token padding, the shape of probs and indices is [tokens, top_k], indicating the selected experts for each token.
        (2) If there's token padding, the shape of probs and indices is [num_expert, capacity], indicating the tokens selected for each expert.
    """
    assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
    num_tokens, num_experts = logits.shape

    def compute_topk(scores, topk, limited_devices=None):
        if limited_devices:
            return device_limited_topk(scores, topk, num_tokens, num_experts, limited_devices)
        else:
            return torch.topk(scores, k=topk, dim=1)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            probs, top_indices = compute_topk(scores, topk, moe_router_topk_limited_devices)
        else:
            scores, top_indices = compute_topk(logits, topk, moe_router_topk_limited_devices)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, moe_router_topk_limited_devices)
            scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        else:
            scores, top_indices = compute_topk(scores, topk, moe_router_topk_limited_devices)
        probs = scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    # norm gate to sum 1
    if topk > 1 and norm_topk_prob:
        denominator = probs.sum(dim=-1, keepdim=True) + 1e-20
        probs = probs / denominator

    if moe_router_topk_scaling_factor:
        probs = probs * moe_router_topk_scaling_factor

    if capacity_factor is None:
        # TopK without capacity , back to core 0.7.0 for better performance
        tokens_per_expert = torch.histc(top_indices, bins=num_experts, min=0, max=num_experts)
        return probs, top_indices, tokens_per_expert
    else:
        # TopK with capacity
        expert_capacity = get_capacity(
            num_tokens=num_tokens * topk,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )
        # TopK selection, Maskout unused experts
        topk_masked_gates = torch.zeros_like(logits).scatter(1, top_indices, probs)
        topk_mask = torch.zeros_like(logits).scatter(1, top_indices, 1)

        # Maskout exceeded tokens
        if drop_policy == "probs":
            capacity_probs, capacity_indices = torch.topk(
                topk_masked_gates, k=expert_capacity, dim=0, sorted=False
            )
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1)
        elif drop_policy == "position":
            _, capacity_indices = torch.topk(topk_mask, k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1)
            capacity_probs = torch.gather(topk_masked_gates, 0, capacity_indices)
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

        if pad_to_capacity:
            final_probs, final_indices = (
                capacity_probs.T.contiguous(),
                capacity_indices.T.contiguous(),
            )
            tokens_per_expert_before_capacity = topk_mask.sum(dim=0)
        else:
            # Get exceed mask and maskout exceeded probs and indices
            final_mask = torch.logical_and(topk_mask, capacity_mask)
            drop_mask = torch.logical_not(final_mask)
            exceed_mask = torch.gather(drop_mask, 1, top_indices)
            final_probs = probs * torch.logical_not(exceed_mask)
            final_indices = top_indices.clone().masked_fill_(
                exceed_mask, torch.iinfo(torch.long).max
            )
            tokens_per_expert_before_capacity = topk_mask.sum(dim=0)
        return final_probs, final_indices, tokens_per_expert_before_capacity


def track_moe_metrics_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        _args = get_args()
        if _args.moe_router_load_balancing_type in ["noaux_tc"]:
            return
        fn(self, *args, **kwargs)

    return wrapper


def get_updated_expert_bias(tokens_per_expert, expert_bias, expert_bias_update_rate):
    """Update expert bias for biased expert routing. See https://arxiv.org/abs/2408.15664v1#
    Args:
        tokens_per_expert (torch.Tensor): The number of tokens assigned to each expert.
        expert_bias (torch.Tensor): The bias for each expert.
        expert_bias_udpate_rate (float): The update rate for the expert bias.
    """
    with torch.no_grad():
        # All Reduce Across TPxCPxDP group
        torch.distributed.all_reduce(
            tokens_per_expert,
            group=parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True),
        )
        average_tokens = tokens_per_expert.sum(dim=-1, keepdim=True) / tokens_per_expert.shape[-1]
        offset = average_tokens - tokens_per_expert
        updated_expert_bias = expert_bias + torch.sign(offset) * expert_bias_update_rate
        return updated_expert_bias
