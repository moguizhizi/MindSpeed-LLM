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

import math
import torch
from megatron.core.transformer.moe.moe_utils import get_capacity


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


def topk_softmax_with_capacity(
    logits: torch.Tensor,
    topk: int,
    capacity_factor: float = None,
    pad_to_capacity: bool = False,
    drop_policy: str = "probs",
    use_pre_softmax: bool = False,
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
    num_tokens = logits.shape[0]
    num_experts = logits.shape[1]
    if use_pre_softmax:
        # Pre softmax
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        probs, top_indices = torch.topk(scores, k=topk, dim=1)
    else:
        # Post softmax
        if topk == 1:
            # Requires applying softmax before selecting the top-k when k is 1, since softmax on a [num_tokens, 1] would yield a zero gradient.
            raise ValueError("Please use --moe-router-pre-softmax when topk is 1.")
        scores, top_indices = torch.topk(logits, k=topk, dim=1)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)

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