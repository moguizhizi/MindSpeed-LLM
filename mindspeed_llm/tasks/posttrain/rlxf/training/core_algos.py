# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch

import mindspeed_llm.tasks.posttrain.rlxf.utils.torch_functional as F
from mindspeed_llm.tasks.posttrain.rlxf.utils.protocol import DataProto


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = F.masked_whiten(advantages, eos_mask)
        advantages = torch.masked_fill(advantages, ~eos_mask, 0)
    return advantages, returns


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if penalty == "kl":
        return logprob - ref_logprob

    if penalty == "abs":
        return (logprob - ref_logprob).abs()

    if penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    if penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def find_first_eos_index(tensor, eos_token_id):
    """
    找到张量中每一行第一个等于 eos_token_id 的索引。

    Args:
        tensor (torch.Tensor): 输入的张量，形状为 (batch_size, seq_len)。

    Returns:
        torch.Tensor: 每一行中每一行第一个等于 eos_token_id 的索引，形状为 (batch_size,)。
                     如果没有找到，返回 -1。
    """

    is_eos = (tensor == eos_token_id)

    # 使用 torch.argmax 找到第一个等于 eos_id 的索引
    score_first_eos_index = torch.argmax(is_eos.int(), dim=1)
    reward_first_eos_index = torch.argmax(is_eos.int(), dim=1) + 1
    max_id = is_eos.shape[1] - 1
    reward_first_eos_index = torch.min(reward_first_eos_index, torch.tensor(max_id, device=reward_first_eos_index.device))
    has_eos = is_eos.any(dim=1)
    score_first_eos_index[~has_eos] = -1
    reward_first_eos_index[~has_eos] = -1

    return score_first_eos_index, reward_first_eos_index


def apply_kl_penalty(config, data: DataProto, kl_ctrl: AdaptiveKLController):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['rm_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]
    eos_token_id = data.meta_info['eos_token_id']
    contain_eos_token = torch.any(responses == eos_token_id, dim=-1)
    if config.algorithm.missing_eos_penalty is not None:
        token_level_scores[~contain_eos_token] -= config.algorithm.missing_eos_penalty
        data.batch['rm_scores'] = token_level_scores
    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    penalty=config.algorithm.kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)
    actual_start = torch.arange(token_level_scores.size(0), device=token_level_scores.device)
    score_first_eos_index, reward_first_eos_index = find_first_eos_index(responses, eos_token_id)
    token_level_rewards = - beta * kld
    token_level_rewards[[actual_start, reward_first_eos_index]] += token_level_scores[[actual_start, score_first_eos_index]]

    current_kl = F.masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, gamma, lam, adv_estimator):
    values = data.batch['values']
    responses = data.batch['responses']
    response_length = responses.size(1)
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]
    token_level_rewards = data.batch['token_level_rewards']

    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        advantages, returns = compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                           values=values,
                                                           eos_mask=response_mask,
                                                           gamma=gamma,
                                                           lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def compute_data_metrics(batch):
    # TODO: add response length
    sequence_score = batch.batch['rm_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    response_length = batch.batch['responses'].shape[-1]

    advantages = batch.batch['advantages']
    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    returns = batch.batch['returns']
    values = batch.batch['values']

    metrics = {
        # score
        'critic/score/mean': torch.mean(sequence_score).detach().item(),
        'critic/score/max': torch.max(sequence_score).detach().item(),
        'critic/score/min': torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean': torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max': torch.max(sequence_reward).detach().item(),
        'critic/rewards/min': torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean': F.masked_mean(advantages, response_mask).detach().item(),
        'critic/advantages/max': torch.max(advantages[response_mask]).detach().item(),
        'critic/advantages/min': torch.min(advantages[response_mask]).detach().item(),
        # returns
        'critic/returns/mean': F.masked_mean(returns, response_mask).detach().item(),
        'critic/returns/max': torch.max(returns[response_mask]).detach().item(),
        'critic/returns/min': torch.min(returns[response_mask]).detach().item(),
        # values
        'critic/values/mean': F.masked_mean(values, response_mask).detach().item(),
        'critic/values/max': torch.max(values[response_mask]).detach().item(),
        'critic/values/min': torch.min(values[response_mask]).detach().item(),
        # response length
        'response_length/mean': torch.mean(response_length).detach().item(),
        'response_length/max': torch.max(response_length).detach().item(),
        'response_length/min': torch.min(response_length).detach().item(),
        # prompt length
        'prompt_length/mean': torch.mean(prompt_length).detach().item(),
        'prompt_length/max': torch.max(prompt_length).detach().item(),
        'prompt_length/min': torch.min(prompt_length).detach().item(),
    }
    return metrics


def compute_data_online_dpo_metrics(batch):
    sequence_score = batch.batch['rm_scores'].sum(-1)
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()

    metrics = {
        # score
        'reward/score/mean': torch.mean(sequence_score).detach().item(),
        'reward/score/max': torch.max(sequence_score).detach().item(),
        'reward/score/min': torch.min(sequence_score).detach().item(),
        # response length
        'response_length/mean': torch.mean(response_length).detach().item(),
        'response_length/max': torch.max(response_length).detach().item(),
        'response_length/min': torch.min(response_length).detach().item(),
        # prompt length
        'prompt_length/mean': torch.mean(prompt_length).detach().item(),
        'prompt_length/max': torch.max(prompt_length).detach().item(),
        'prompt_length/min': torch.min(prompt_length).detach().item(),
    }
    return metrics
