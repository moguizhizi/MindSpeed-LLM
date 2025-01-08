### 权重转换特性清单

MindSpeed-LLM 支持 Huggingface、Megatron-Legacy 以及 Megatron-Core 之间的权重格式互转，具体功能列表如下:

<table>
  <thead>
    <tr>
      <th>源格式</th>
      <th>目标格式</th>
      <th>支持特性</th>
      <th>特性入参</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="12">HuggingFace </td>
      <td rowspan="4">Megatron-Legacy</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行动态划分</td>
      <td>--num-layer-list</td>
    </tr>
    <tr>
      <td>虚拟流水并行</td>
      <td>--num-layers-per-virtual-pipeline-stage</td>
    </tr>
    <tr>
      <td rowspan="8">Megatron-Core</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行动态划分</td>
      <td>--num-layer-list</td>
    </tr>
    <tr>
      <td>虚拟流水并行</td>
      <td>--num-layers-per-virtual-pipeline-stage</td>
    </tr>
    <tr>
      <td>专家并行</td>
      <td>--target-expert-model-parallel-size</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="20">Megatron-Legacy </td>
      <td rowspan="6">Huggingface</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>LoRA训练模块</td>
      <td>--lora-target-modules</td>
    </tr>
    <tr>
      <td>LoRA权重</td>
      <td>--lora-load</td>
    </tr>
    <tr>
      <td>LoRA r</td>
      <td>--lora-r</td>
    </tr>
    <tr>
      <td>LoRA alpa</td>
      <td>--lora-alpha</td>
    </tr>
    <tr>
      <td rowspan="4">Megatron-Core</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行动态划分</td>
      <td>--num-layer-list</td>
    </tr>
    <tr>
      <td>虚拟流水并行</td>
      <td>--num-layers-per-virtual-pipeline-stage</td>
    </tr>
    <tr>
      <td rowspan="6">Megatron-Legacy</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>LoRA训练模块</td>
      <td>--lora-target-modules</td>
    </tr>
    <tr>
      <td>LoRA权重</td>
      <td>--lora-load</td>
    </tr>
    <tr>
      <td>LoRA r</td>
      <td>--lora-r</td>
    </tr>
    <tr>
      <td>LoRA alpa</td>
      <td>--lora-alpha</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="13">Megatron-Core </td>
      <td rowspan="2">Huggingface</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td rowspan="4">Megatron-Legacy</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行动态划分</td>
      <td>--num-layer-list</td>
    </tr>
    <tr>
      <td>虚拟流水并行</td>
      <td>--num-layers-per-virtual-pipeline-stage</td>
    </tr>
    <tr>
      <td rowspan="5">Megatron-Core</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>专家并行</td>
      <td>--target-expert-model-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行动态划分</td>
      <td>--num-layer-list</td>
    </tr>
    <tr>
      <td>虚拟流水并行</td>
      <td>--num-layers-per-virtual-pipeline-stage</td>
    </tr>
  </tbody>
</table>

### 权重转换使用

#### 1. 权重下载

从Huggingface等网站下载开源模型权重

预训练权重链接在 [稠密模型](../models/dense_model.md)和[MoE模型](../models/moe_model.md) 章节列表的`参数`列链接中可以获取；更多社区资源可以在`模型`列链接中获取，如`Chat/Instruct`权重等。

权重可以基于网页直接下载，也可以基于命令行下载，保存到MindSpeed-LLM/model_from_hf目录，比如：


```shell
#!/bin/bash
mkdir ./model_from_hf/llama-2-7b-hf/
cd ./model_from_hf/llama-2-7b-hf/
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/config.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/generation_config.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model-00001-of-00002.bin
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model-00002-of-00002.bin
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model.bin.index.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/special_tokens_map.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.model
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer_config.json
cd ../../
```

#### 2. 权重转换

##### 2.1 Huggingface权重转换到Megatron-LM格式

```shell
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 4 \
    --num-layer-list 8,8,8,8 \
    --model-type-hf llama2 \
    --use-mcore-models \
    --load-dir ./model_from_hf/llama-2-7b-hf/ \
    --save-dir ./model_weights/llama-2-7b-mcore/ \
    --tokenizer-model ./model_from_hf/llama-2-7b-hf/tokenizer.model
```
<table>
  <thead>
    <tr>
      <th>参数</th>
      <th>说明</th>
      <th>可选/必选</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>--target-tensor-parallel-size</td>
      <td>TP 切分数量，默认为 1</td>
      <td>必选</td>
    </tr>
    <tr>
      <td>--target-pipeline-parallel-size</td>
      <td>PP 切分数量，默认为 1</td>
      <td>必选</td>
    </tr>
    <tr>
      <td>--num-layer-list</td>
      <td>动态PP划分，通过列表指定每个PP Stage的层数，默认为None</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--num-layers-per-virtual-pipeline-stage</td>
      <td>VPP划分，指定VPP的每个Stage层数，默认为None</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--use-mcore-models</td>
      <td>转换为Megatron-Mcore权重，若不指定，则默认转换为Megatron-Legacy权重</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--model-type-hf</td>
      <td>huggingface模型类别，默认为llama2</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--tokenizer-model</td>
      <td>需要指明到具体的分词器模型文件，如 tokenizer.model、tokenizer.json、qwen.tiktoken、None等，具体取决于huggingface中词表文件的格式形式</td>
      <td>必选</td>
    </tr>
    <tr>
      <td>--params-dtype</td>
      <td>指定权重转换后的权重精度模式，默认为fp16，如果源格式文件为bf16，则需要对应设置为bf16，影响推理或评估结果</td>
      <td>必选</td>
    </tr>
  </tbody>
</table>


**注意**：
1、VPP和动态PP划分只能二选一

2、目前支持的模型见 [model_cfg.json](https://gitee.com/ascend/MindSpeed-LLM/blob/master/modellink/tasks/checkpoint/model_cfg.json)


【启动脚本】

MindSpeed-LLM Huggingface到Megatron-Legacy权重转换脚本命名风格及启动方法为：

```shell
# 命名及启动：
# bash examples/legacy/model_name/ckpt_convert_xxx_hf2legacy.sh
# 需要配置并行参数以及权重词表加载保存等路径

bash examples/legacy/llama2/ckpt_convert_llama2_hf2legacy.sh
```

MindSpeed-LLM Huggingface到Megatron-Mcore权重转换脚本命名风格及启动方法为：

```shell
# 命名及启动：
# bash examples/mcore/model_name/ckpt_convert_xxx_hf2mcore.sh
# 需要配置并行参数以及权重词表加载保存等路径

bash examples/mcore/llama2/ckpt_convert_llama2_hf2mcore.sh
```

##### 2.2 Megatron-LM权重转换到Huggingface格式

```shell
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --model-type-hf llama2 \
    --use-mcore-models \
    --load-dir ./model_weights/llama-2-7b-mcore/ \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_from_hf/llama-2-7b-hf/  # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-2-7b-hf/mg2hf/
```

参数意义参考2.1

【启动脚本】

MindSpeed-LLM Megatron-Legacy到Huggingface的权重转换脚本命名风格及启动方法为：

```shell
# 命名及启动：
# bash examples/legacy/model_name/ckpt_convert_xxx_legacy2hf.sh
# 需要配置并行参数以及权重词表加载保存等路径

bash examples/legacy/llama2/ckpt_convert_llama2_legacy2hf.sh
```

MindSpeed-LLM Megatron-Mcore到Huggingface的权重转换脚本命名风格及启动方法为：

```shell
# 命名及启动：
# bash examples/mcore/model_name/ckpt_convert_xxx_mcore2hf.sh
# 需要配置并行参数以及权重词表加载保存等路径

bash examples/mcore/llama2/ckpt_convert_llama2_mcore2hf.sh
```

##### 2.3 Megatron-LM格式权重互转

```shell
# legacy转legacy
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_weights/llama-2-7b-legacy/ \
    --save-dir ./model_weights/llama-2-7b-legacy_tp2pp2/

# legacy转mcore
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --use-mcore-models \
    --load-from-legacy \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_weights/llama-2-7b-legacy/ \
    --save-dir ./model_weights/llama-2-7b-mcore_tp2pp2/

# mcore转mocre
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --use-mcore-models \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_weights/llama-2-7b-mcore/ \
    --save-dir ./model_weights/llama-2-7b-mcore_tp2pp2/

# mcore转legacy
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --use-mcore-models \
    --save-to-legacy \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_weights/llama-2-7b-mcore/ \
    --save-dir ./model_weights/llama-2-7b-legacy_tp2pp2/
```

【--load-from-legacy】 

legacy转mcore时设置此参数以指定导入权重格式为legacy

【--save-to-legacy】 

mcore转legacy时设置此参数以指定保存权重格式为legacy

其余参数意义参考2.1

注：上述权重legacy和mcore互转为高阶功能，MindSpeed-LLM基于llama2提供基础能力，并进行版本迭代看护，其余模型的支持需要用户自行修改支持

##### 2.4 lora权重与base权重合并

在上述权重转换命令中，加入如下参数可以将训练的lora权重与base进行融合。

```bash
--lora-load ./ckpt/llama-2-7b-lora  \
--lora-r 16 \
--lora-alpha 32 \
--lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
```

<table>
  <thead>
    <tr>
      <th>参数</th>
      <th>说明</th>
      <th>可选/必选</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>--lora-load</td>
      <td>加载 lora 微调后生成的权重</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--lora-r</td>
      <td>LoRA中的秩（rank），它决定了低秩矩阵的大小</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--lora-alpha</td>
      <td>定义了LoRA适应的学习率缩放因子。这个参数影响了低秩矩阵的更新速度</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--lora-target-modules</td>
      <td>定义了Lora目标模块，字符串列表，由空格隔开，无默认值。每一个字符串是需要进行LoRA微调的层的名称。</td>
      <td>可选</td>
    </tr>
  </tbody>
</table>

【合并后转换为Megatron-Legacy权重】

```shell
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --load-dir ./model_weights/llama-2-7b-legacy/ \
    --lora-load ./ckpt/llama-2-7b-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_weights/llama-2-7b-lora2legacy
```


转换脚本命名风格及启动方法为：

```shell
# 命令启动方式以 legacy 下的模型为例子
bash examples/legacy/llama2/ckpt_convert_llama2_legacy2legacy_lora.sh
```

【合并后转换为Huggingface权重】

```shell
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./model_weights/llama-2-7b-legacy/ \
    --lora-load ./ckpt/llama-2-7b-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_from_hf/llama-2-7b-hf/    # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-2-7b-hf/mg2hg/
```

转换脚本命名风格及启动方法为：

```shell
# 命令启动方式以 legacy 下的模型为例子
bash examples/legacy/llama2/ckpt_convert_llama2_legacy2hf_lora.sh
```

**注意：** lora参数值需与lora微调时的参数保持一致