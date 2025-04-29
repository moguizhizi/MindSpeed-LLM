---
license: mit
language:
  - zh
  - en
frameworks:
  - PyTorch
pipeline_tag: text-generation
hardwares:
  - NPU
---

# 0day首发！MindSpeed-LLM适配Qwen3并上线开源社区

Qwen3是阿里云于2025年4月29日发布并开源，作为 Qwen 系列中的最新一代大型语言模型，提供了一系列密集型和混合专家（MoE）模型。本次Qwen发布多个尺寸模型，覆盖235B/32B/30B/14B/8B/4B/1.7B/0.6B。在Qwen-3正式发布的同一天，MindSpeed-LLM便立刻支持该模型的完美跑通，标志着MindSpeed-LLM在大规模语言模型应用和高效部署方面的技术优势再次得到巩固。

## MindSpeed-LLM：为Qwen-3赋能，极速支持无缝集成

MindSpeed-LLM作为昇腾AI生态的重要技术支撑，专为大规模语言模型设计，具有超强的计算能力和灵活的开发支持。随着Qwen-3的发布，MindSpeed-LLM已立即做好了全面支持和优化准备，为开发者提供了一个稳定、高效的平台来快速部署和调优Qwen-3模型。

- 硬件与框架深度协同，立刻跑通：MindSpeed-LLM与昇腾芯片的深度集成，使得Qwen-3大语言模型在发布的第一时间内，就能够顺利跑通并高效运行。无论是在训练过程中，还是在推理阶段，MindSpeed-LLM都为Qwen-3提供了最佳的硬件加速支持，确保性能的最大化释放。

- 开箱即用，无需复杂配置：开发者只需简单配置，即可在MindSpeed-LLM上无缝运行Qwen-3模型。框架提供了完整的工具链，帮助开发者快速将Qwen-3应用到实际项目中，减少了复杂的调优过程，缩短了开发周期。

- 分布式计算优化：MindSpeed-LLM内置的分布式计算能力，能够有效利用多台昇腾AI硬件，确保Qwen-3在大规模并发任务下的稳定运行，极大提升了处理效率和响应速度。

MindSpeed-LLM框架与Qwen-3的同步发布并立刻支持跑通，标志着昇腾平台在大语言模型领域的技术实力再次提升。开发者可以在第一时间内，借助强大的昇腾计算能力，快速将Qwen-3应用于实际项目，进一步加速智能应用的落地。

## 模型支持列表

| 模型名称|功能|Released|
|:---:|:---:|:---:|
| Qwen3 | OK | Doing |

## 魔乐社区链接
https://modelers.cn/models/MindSpeed/Qwen3-0.6B-Base

https://modelers.cn/models/MindSpeed/Qwen3-8B-Base

https://modelers.cn/models/MindSpeed/Qwen3-4B-Base

https://modelers.cn/models/MindSpeed/Qwen3-1.7B-Base

https://modelers.cn/models/MindSpeed/Qwen3-30B-A3B-Base

https://modelers.cn/models/MindSpeed/Qwen3-14B-Base
https://modelers.cn/models/MindSpeed/Qwen3-8B
https://modelers.cn/models/MindSpeed/Qwen3-4B

https://modelers.cn/models/MindSpeed/Qwen3-32B

https://modelers.cn/models/MindSpeed/Qwen3-30B-A3B

https://modelers.cn/models/MindSpeed/Qwen3-235B-A22B

https://modelers.cn/models/MindSpeed/Qwen3-1.7B

https://modelers.cn/models/MindSpeed/Qwen3-14B

https://modelers.cn/models/MindSpeed/Qwen3-0.6B



# MindSpeed跑通Qwen-3模型手把手教程：快速部署与应用
本教程将引导您完成从环境准备到模型训练的完整过程，确保您能够快速上手并顺利跑通Qwen-3模型。我们提供详细的步骤说明，帮助您在MindSpeed框架下实现Qwen-3模型的无缝运行。


###  目录

- [环境配置](#环境配置)
  - [硬件要求](#硬件要求)
  - [MindSpeed-LLM仓库部署](#MindSpeed-LLM仓库部署)
- [权重转换](##权重转换)
- [数据预处理](##数据预处理)
- [模型训练](##训练)
- [推理](##推理)
- [评估](##评估)



# 环境配置

## 硬件要求

qwen3的参考硬件配置如下,本文将以A2 单机8卡训练和推理为例进行介绍：

|类型|  硬件 |       配置        |
|:---:|:---:|:---------------:|
|全参微调| NPU | 8 x Ascend NPUs |

## MindSpeed-LLM仓库部署

MindSpeed-LLM的主要依赖配套如下表，安装步骤参考[安装指导](https://gitee.com/ascend/MindSpeed-LLM/blob/master/docs/features/install_guide.md)。

<table>
  <tr>
    <th>依赖软件</th>
    <th>版本</th>
  </tr>
  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">商发版本</td>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>CANN Toolkit（开发套件）</td>
      <td rowspan="3">商发版本</td>
  </tr>
  <tr>
    <td>CANN Kernel（算子包）</td>
  </tr>
  <tr>
    <td>CANN NNAL（Ascend Transformer Boost加速库）</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Python</td>
    <td>>=3.10</td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>2.5.1</td>
  </tr>
  <tr>
    <td>torch_npu插件</td>
    <td >2.5.1</td>
  </tr>
  <tr>
    <td>apex</td>
    <td >商发版本</td>
  </tr>
</table>


#### 1. 仓库拉取

```shell
git clone https://gitee.com/ascend/MindSpeed-LLM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../MindSpeed-LLM/
cd ..
cd MindSpeed-LLM
mkdir logs
mkdir dataset
mkdir ckpt
```

<a id="jump1.2"></a>

#### 2. 环境搭建

torch npu 与 CANN包参考链接：[安装包参考链接](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
pip install torch-2.1.0-cp310-cp310m-manylinux2014_aarch64.whl
pip install torch_npu-2.1.0*-cp310-cp310m-linux_aarch64.whl

# apex for Ascend 参考 https://gitee.com/ascend/apex
# 建议从原仓编译安装

# 安装加速库
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.8.0
git checkout 2c085cc9
pip install -r requirements.txt
pip3 install -e .
cd ../MindSpeed-LLM


# 安装其余依赖库
pip install -e .
```



## 权重转换

1. 权重下载 

    从 [huggingface](https://huggingface.co/THUDM/GLM-4-9B-0414) 或者魔乐社区[链接](https://modelers.cn/models/zhipuai/GLM-4-9B-0414)下载权重和配置文件

2. 权重转换

    MindSpeed-LLM提供[脚本](https://gitee.com/lliilil/MindSpeed-LLM/blob/master/tests/0day/glm4-0414/ckpt_convert_glm4_0414_hf2mcore.sh)将已经huggingface开源权重转换为mcore权重，用于训练、推理、评估等任务。

    使用方法如下，请根据实际需要的TP/PP等切分策略和权重路径修改权重转换脚本
    ```sh
    cd MindSpeed-LLM
    bash tests/0day/glm4-0414/ckpt_convert_glm4_0414_hf2mcore.sh
      ```



## 数据预处理

MindSpeed-LLM提供[脚本](https://gitee.com/lliilil/MindSpeed-LLM/blob/master/tests/0day/glm4-0414/data_convert_glm4_0414_pretrain.sh) 进行数据集处理

使用方法如下，请根据实际需要修改以下参数
```sh
cd MindSpeed-LLM
bash tests/0day/qwen3/data_convert_qwen3_pretrain.sh
  ```
  
| 参数名  | 含义                |
|--------|-----------------|
| --input | 数据集路径  |
| --tokenizer-name-or-path | 模型tokenizer目录    |
| --output-prefix | 数据集处理完的输出路径及前缀名  |



## 训练

 ```sh
cd MindSpeed-LLM
 bash tests/0day/qwen3/pretrain_qwen3_8k_ptd.sh
   ```
用户需要根据实际情况修改脚本中以下变量
  | 变量名  | 含义                |
  |--------|-----------------|
  | MASTER_ADDR | 多机情况下主节点IP  |
  | NODE_RANK | 多机下，各机对应节点序号    |
  | CKPT_SAVE_DIR | 训练中权重保存路径  |
  | DATA_PATH | 数据预处理后的数据路径  |
  | TOKENIZER_PATH | qwen3 tokenizer目录  |
  | CKPT_LOAD_DIR | 权重转换保存的权重路径，为初始加载的权重，如无初始权重则随机初始化  |

## 推理

 ```sh
cd MindSpeed-LLM
 bash tests/0day/qwen3/generate_qwen3_ptd.sh
   ```
用户需要根据实际情况修改脚本中以下变量
  | 变量名  | 含义                |
  |--------|-----------------|
  | MASTER_ADDR | 多机情况下主节点IP  |
  | NODE_RANK | 多机下，各机对应节点序号    |
  | CHECKPOINT | 训练保存的权重路径  |
  | TOKENIZER_PATH | qwen3 tokenizer目录  |

## 评估

 ```sh
cd MindSpeed-LLM
 bash tests/0day/qwen3/evaluate_qwen3_ptd.sh
   ```
用户需要根据实际情况修改脚本中以下变量
  | 变量名  | 含义                |
  |--------|-----------------|
  | MASTER_ADDR | 多机情况下主节点IP  |
  | NODE_RANK | 多机下，各机对应节点序号    |
  | CKPT_SAVE_DIR | 训练中权重保存路径  |
  | DATA_PATH | 数据预处理后的数据路径  |
  | TOKENIZER_PATH | qwen3 tokenizer目录  |
  | CKPT_LOAD_DIR | 权重转换保存的权重路径，为初始加载的权重，如无初始权重则随机初始化  |

