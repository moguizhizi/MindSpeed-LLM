# rejection sampling算法

rejection sampling拒绝采样算法实现参考了[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
库的拒绝采样算法，不同点有：
- 1）增加了具有参考答案的数学类问题的支持
- 2）支持根据正确答案进行规则过滤
- 3）判别式结果奖励模型（ORM）替换为判别效果更好的生成式奖励模型（GenRM），GenRM使用vllm进行评分。 

rejection sampling拒绝采样算法由两部分组成：
- 1）step1：基于vllm的批量回答生成，给定一个prompt，LLM模型输出N个回答。
- 2）step2：通过奖励模型对每个回答进行打分，筛选得到得分最高的回答。


## 使用说明


### Step1：vllm批量推理示例

除常规推理参数外，vllm批量推理需要额外配置以下参数：
- **`--task generate_vllm`**

  generate_vllm为批量推理阶段，rejection_sampling为奖励模型拒绝采样阶段

- **`--rollout-batch-size`**

  必选，批量推理的prompt数目

- **`--iter`**

  可选，用于对iter*rollout_batch_size:(iter+1)*rollout.batch_size范围内的数据集进行切片，默认0

- **`--max-samples`**

  可选，最大样本数目

- **`--best-of-n`**

  必选，一个prompt推理N次

- **`--pretrain`**

  必选，LLM模型的路径，模型为hf格式

- **`--data`**

  必选，数据集路径

- **`--output-path`**

  必选，结果保存路径， 保存为json格式。

- **`--map-keys`**

  必选，配置字段映射来使用数据集，prompt字段必须，gt_answer字段可选

- **`--apply-chat-template`**

  可选，使用tokenizer进行apply_chat_template

- **`--input-template`**

  可选，可外部输入指定template格式
- 
以 [math_level3to5_data_processed_with_qwen_prompt 数据集](https://huggingface.co/datasets/pe-nlp/math_level3to5_data_processed_with_qwen_prompt) 为例。

```shell
POLICY_MODEL_PATH=/Qwen2.5-7B-Instruct
DATA_PATH="parquet@/data/pe-nlp/math_level3to5_data_processed_with_qwen_prompt"

ROLLOUT_BATCH_SIZE=1000
N=8
iter=0

python mindspeed_llm/tasks/posttrain/rejection_sampling/rejection_sampling.py \
   --pretrain $POLICY_MODEL_PATH \
   --task generate_vllm \
   --max-new-tokens 2048 \
   --prompt-max-len 2048 \
   --dataset $DATA_PATH \
   --map-keys '{"prompt":"input","gt_answer":"gt_answer","response":""}' \
   --temperature 0.7 \
   --repetition-penalty 1.05 \
   --top-p 0.8 \
   --best-of-n $N \
   --enable-prefix-caching \
   --tp-size 4 \
   --iter $iter \
   --rollout-batch-size $ROLLOUT_BATCH_SIZE \
   --output-path generate_output.jsonl
```


### Step2：GenRM生成式奖励模型拒绝采样

除常规推理参数外，GenRM生成式奖励模型拒绝采样需要额外配置以下参数：

- **`--task rejection_sampling`**

  generate_vllm为批量推理阶段，rejection_sampling为奖励模型拒绝采样阶段

- **`--map-keys`**

  必选，配置字段映射来使用数据集，prompt、response字段必须，gt_answer字段可选

- **`--use-ground-truth-answer`**

  可选，使用正确答案来评估得分

- **`--use-rules`**

  可选，使用规则来过滤错误答案的样本

- **`--pretrain`**

  必选，LLM模型的路径，模型为hf格式

- **`--data`**

  必选，数据集路径，即step1输出的路径

- **`--output-path`**

  必选，结果保存路径。 保存为json格式，有prompt,output,reward三个字段。


```shell
POLICY_MODEL_PATH=/Qwen2.5-7B-Instruct
data="generate_output.jsonl"

python mindspeed_llm/tasks/posttrain/rejection_sampling/rejection_sampling.py \
   --pretrain $POLICY_MODEL_PATH \
   --task rejection_sampling \
   --map-keys '{"prompt":"prompt","gt_answer":"gt_answer","response":"output"}' \
   --use-ground-truth-answer \
   --max-new-tokens 2048 \
   --prompt-max-len 2048 \
   --dataset $data \
   --temperature 0.3 \
   --top-p 0.3 \
   --repetition-penalty 1.05 \
   --enable-prefix-caching \
   --tp-size 4 \
   --output-path rejection_sampling_output.jsonl

```

## 备注
该算法使用vllm进行推理，vllm版本为华为云npu版本，请联系华为云vllm团队进行支持。

## 参考文献

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)