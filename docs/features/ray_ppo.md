# 后训练方法 Ray PPO

[PPO（Proximal Policy Optimization）](https://arxiv.org/abs/1707.06347)是一种强化对齐微调方法，常用于人类反馈强化学习（Reinforcement Learning with Human Feedback, RLHF）任务。

PPO方法中包含了四个模型：Actor，Critic，Reference，Reward。其中Actor/Reference模型是经过预训练和指令微调（Supervised Fine-Tuning，SFT）得到的大语言模型，Critic和Reward是训练得到的奖励模型。PPO 的训练目标是使得 Actor 模型的回答可以更加符合人类偏好。

# 使用说明

## 环境配置

配置MindSpeed-LLM基础环境: 参考[MindSpeed-LLM 使用指南-前期准备](https://gitee.com/ascend/MindSpeed-LLM/blob/master/docs/USER_GUIDE.md#%E5%89%8D%E6%9C%9F%E5%87%86%E5%A4%87)

## 数据预处理

数据集转换参考脚本：MindSpeed-LLM\examples\mcore\llama3\data_convert_llama3_ppo.sh
以 [descriptiveness 数据集](https://huggingface.co/datasets/trl-internal-testing/descriptiveness-sentiment-trl-style/tree/main/data) 为例。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./dataset/llama3-hf/

python ./preprocess_data.py \
    --input ./dataset/descriptiveness-00000-of-00001.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama3-hf/ \
    --output-prefix ./dataset/llama3-hf/descriptiveness \
    --workers 16 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name  PPOAlpacaStyleInstructionHandler \
    --prompt-type llama3 \
    --map-keys '{"prompt":"prompt", "query":"", "response": "prompt", "system":""}'
```

## 模型权重转换

根据 PPO 算法要求，Actor 和 Reference 模型应该使用 SFT 微调后的模型进行初始化，Critic 和 Reward 模型应该使用奖励模型训练后的模型进行初始化。PPO算法模型权重均使用Megatron-mcore格式，其他格式的权重需要进行模型权重转换，具体可参考[MindSpeed-LLM 使用指南-权重转换](https://gitee.com/ascend/MindSpeed-LLM/blob/master/docs/USER_GUIDE.md#%E6%9D%83%E9%87%8D%E4%B8%8B%E8%BD%BD%E5%8F%8A%E8%BD%AC%E6%8D%A2)。

## 启动方式

### 单机

通过 --config-name 传递选取的 config 文件名（不添加.yaml后缀），可以通过下列命令直接启动训练（Llama32 1B 模型可单机运行）。
目前已支持的配置文件放置在 configs/rlxf/ 文件夹下。配置文件的具体说明见下文。

```bash
python ray_gpt.py --config-name ppo_trainer_llama32_1b
```

### 多机

多机运行程序时，需要首先进入对应目录，并激活conda或docker环境：

```bash
cd MindSpeed-LLM
conda activate xxx
```

然后，在主节点上启动 Ray 集群：

```bash
# 配置最大文件描述符环境变量
ulimit -n 32768
# 创建一个集群，端口6344，dashboard端口8260，有8个NPU
ray start --head --port 6344 --dashboard-host=0.0.0.0 --dashboard-port=8260 --resources='{"NPU": 8}'
```

随后，在其他节点加入主节点的集群

```bash
# 配置最大文件描述符环境变量
ulimit -n 32768
# IP_ADDRESS 处填写主节点 IP 地址
ray start --address="IP_ADDRESS:6344" --resources='{"NPU": 8}'
```

在完成 Ray 集群构建后，在主节点启动运行程序即可（Llama3 8B 模型可双机运行）

```bash
python ray_gpt.py --config-name ppo_trainer_llama3_8b
```

## 配置文件

由于 PPO 训练过程中涉及 4 个模型，通过将模型参数和训练配置解耦的层级化参数配置，来简化 PPO 训练的参数配置过程。RLXF 训练涉及到的所有配置文件均存储在 configs/rlxf 路径下，其中 model 文件夹下存储了模型结构相关的配置文件，PPO训练相关的模型参数文件以ppo_trainer_{模型名}.yaml方式命名。

在每个 ppo_trainer 配置文件中，需要包含defaults，training，resource_pool，algorithm等字段，以及 PPO 训练过程中涉及到的 4 个角色 actor，critic，reward，ref的配置。其中：

1. defaults 负责引入模型配置文件，在 defaults 中应列举本配置文件中所需要用到的所有模型配置，模型配置可以在下方四个角色的具体配置中通过 model 字段进行选择。
2. training 字段设置的参数为所有 4 个角色通用的默认参数，这些参数可以在下方进一步被角色的单独配置所覆盖。
3. resource_pool 字段指定了各个角色所需的 NPU 资源数量。
4. algorithm 字段配置计算PPO中advantages算法的相关参数。
5. actor，critic，reward，ref 字段分别指定了PPO算法中四个角色训练相关的参数配置。

## 参数解析

相较于普通模型训练，PPO增加一些特殊参数：

### `training:`

* `stage`：用于指定训练算法，使用 Ray PPO 训练须设置为`ray_ppo`;

### `actor_rollout:`

* `do_sample`：控制 Actor 模型进行推理时是否采样，默认为 False ；
* `ppo_mini_batch_size`：Actor 模型的 mini_batch_size，默认为1；
* `max_prompt_length`：PPO 训练中最大 prompt 长度，默认为512；
* `num_samples_per_step`：Actor 推理时每个step的推理样本数量，默认为1；
* `ppo_epochs`：Actor 训练对同一批经验数据的重复次数，默认为1；
* `clip_ratio`：Actor模型训练计算损失函数时的clip比例，默认为0.2；
* `shuffle_minibatch`：Actor 训练时是否对 minibatch 进行 shuffle，默认为 False；
* `num_gpus_for_train` ：Actor 模型分配给训练部分的显卡数量；
* `num_gpus_for_infer` ：Actor 模型分配给推理部分的显卡数量；

### `critic:`

* `cliprange_value`：Critic 模型计算损失函数时 clip 范围，默认为3.0；
* `critic_mini_batch_size`：Critic 模型设置的 mini_batch_size，默认为1；
* `critic_update_epochs`：Critic 训练对同一批经验数据的重复次数，默认为1；

### `algorithm:`

* `adv_estimator`：advantages计算的方式，通常采用gae(广义优势估计Generalized Advantage Estimation, GAE);
* `gamma`：计算 advantage 时的折扣因子；
* `lam`：GAE 优势计算的 lambda 值；
* `kl_penalty`：KL 散度计算方式；
* `kl_ctrl:`
  * `kl_coef`：施加 KL 散度惩罚的系数；
  * `type`：KL 散度惩罚的系数类型；
* `missing_eos_penalty`：缺少序列结束符EOS时的惩罚系数；

### `resource_pool:`

* `actor_rollout`：给 Actor 模型训练和推理总共分配的显卡数量；
* `ref`：给 Reference 模型分配的显卡数量；
* `critic`：给 Critic 模型分配的显卡数量；
* `reward`：给 Reward 模型分配的显卡数量；

# 精度对比

我们与 HuggingFace 的强化学习开源仓库 [TRL](https://github.com/huggingface/trl/) 进行了精度对比，来辅助验证算法实现的正确性。为了与基准方法进行精度对齐，在 Actor 推理时采用贪婪（greedy）策略去除随机性，训练过程中的 critic loss和 actor loss对比如下图所示。

<figure>
  <img src="../../sources/images/ray_ppo/llama32_1b_trl_compare_generate_response.png" alt="Image description" />
  <figcaption style="text-align: center;">未固定 responses 时 loss 对比图 (左为 actor loss，右为 critc loss)</figcaption>
</figure>

然而，由于 greedy 方法的策略为选取 logits 最大的 token，当如果两个 token 的 logits 值十分接近时，可能会导致选取的 token 的结果产生偏差。这种误差会被多次迭代逐步累计放大，最终影响到 loss 精度对齐。

因此，我们额外补充了固定 responses 方式进行精度对齐的实验。可以看到，固定 responses 后 loss 能够较好地实现对齐。

<figure>
  <img src="../../sources/images/ray_ppo/llama32_1b_trl_compare_fixed_response.png" alt="Image description" />
  <figcaption style="text-align: center;">固定 responses 后的 loss 对比图 (左为 actor loss，右为 critc loss)</figcaption>
</figure>

注: 为了验证 actor loss 的精度对齐效果，这里并未直接对比 PPO 算法中记录的 actor loss。这是由于 PPO 算法在计算 advantages 时，为保证算法的稳定性，会在 Actor 训练过程中在 minibatch 间做白化操作（将其分布的均值调整为0，方差调整为1）。这导致 Actor 虽然在进行梯度更新时使用每个 minibatch 计算的 loss，但记录下来的 minibatch 间的 loss 均值接近于 0 。因此，我们选择了记录 Actor 每个 minibatch loss 绝对值的均值，来验证精度对齐效果。


# 参考文献

[PPO](https://arxiv.org/abs/1707.06347)

