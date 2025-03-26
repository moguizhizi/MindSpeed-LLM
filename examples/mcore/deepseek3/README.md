## DeepSeek-V3权重转换

### 使用说明

可以将已经反量化为bf16数据格式的[huggingface权重](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/tree/main)转换为mcore权重，用于微调、推理、评估等任务。反量化方法请参考DeepSeek官方提供的[代码](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/inference/fp8_cast_bf16.py)。
并将训练好的megatron mcore格式的权重转换回huggingface格式。

### 启动脚本

使用DeepSeek-V3模型目录下的<a href="../../mcore/deepseek3/ckpt_convert_deepseek3_hf2mcore.sh">huggingface转megatron脚本</a>和<a href="../../mcore/deepseek3/ckpt_convert_deepseek3_mcore2hf.sh">megatron转huggingface脚本</a>。

#### 填写相关参数

【--moe-grouped-gemm】

当每个专家组有多个专家时，可以使用Grouped GEMM功能来提高利用率和性能。

【--target-tensor-parallel-size】

张量并行度，默认值为1。

【--target-pipeline-parallel-size】

流水线并行度，默认值为1。

【--target-expert-parallel-size】

专家并行度，默认值为1。

【--num-layers-per-virtual-pipeline-stage】

虚拟流水线并行，默认值为None, 注意参数--num-layers-per-virtual-pipeline-stage 和 --num-layer-list 不能同时使用。

【--load-dir】

已经反量化为bf16数据格式的huggingface权重。

【--save-dir】

转换后的megatron格式权重的存储路径。

【--num-nextn-predict-layers】

MTP层的层数。如不需要MTP层，可设置为0。最大可设置为1。默认值为0。
MTP层权重默认存储在最后一个pp stage。

【--num-layers】

模型层数，该层数不包含MTP层。默认值为61。如配置空操作层，num-layers的值应为总层数（不包含MTP层）加上空操作层层数。

【--first-k-dense-replace】

moe层前的dense层数，最大可设置为3。默认值为3。

【--num-layer-list】

指定每个pp的层数，相加要等于num-layers。默认值为None。

【--noop-layers】

自定义空操作层。与--num-layer-list互斥，二者选其一使用。默认值为None。

【--moe-tp-extend-ep】

TP拓展EP，专家层TP组不切分专家参数，切分专家数量。默认值为False。

【--mla-mm-split】

在MLA中，将2个up-proj matmul操作拆分成4个。默认值为False。

【--qlora-nf4】

指定是否开启QLoRA权重量化转换，默认为False

#### 合并lora权重和base权重

##### 相关参数

【--load-dir】

指定base权重加载路径

【--lora-load】

指定lora权重加载路径

【--lora-r】

lora矩阵的秩

【--lora-alpha】

缩放因子，缩放低秩矩阵的贡献


##### 注意事项

如果需要合并同一份权重中的lora和base权重

示例：

```
python examples/mcore/deepseek3/convert_ckpt_deepseek3_mcore2hf.py \
    --source-tensor-parallel-size 1 \
    --source-pipeline-parallel-size 4 \
    --source-expert-parallel-size 8 \
    --load-dir ./model_weights/deepseek3-lora \   
    --save-dir ./model_from_hf/deepseek3-hf \
    --num-layers 61 \
    --first-k-dense-replace 3 \
    --num-layer-list 16,15,15,15 \
    --num-nextn-predict-layers 1 \
    --lora-r 8 \
    --lora-alpha 16 \
```

【--load-dir】填写lora权重路径，该权重包括base权重和lora权重

【--lora-r】、【--lora-alpha】与lora微调时配置相同

如果需要合并base权重和独立的lora权重

示例：
```
python examples/mcore/deepseek3/convert_ckpt_deepseek3_mcore2hf.py \
    --source-tensor-parallel-size 1 \
    --source-pipeline-parallel-size 4 \
    --source-expert-parallel-size 8 \
    --load-dir ./model_weights/deepseek3-mcore \
    --lora-load ./ckpt/filter_lora \
    --save-dir ./model_from_hf/deepseek3-hf \
    --num-layers 61 \
    --first-k-dense-replace 3 \
    --num-layer-list 16,15,15,15 \
    --num-nextn-predict-layers 1 \
    --lora-r 8 \
    --lora-alpha 16 \
    # --num-layer-list, --noop-layers, --num-layers-per-virtual-pipeline-stage等参数根据任务需要进行配置
```

【--load-dir】指定base权重路径

【--lora-load】指定lora权重路径，注意该权重仅为lora权重，可以在lora微调中加入'--lora-ckpt-filter'，只保存lora权重

【--lora-r】、【--lora-alpha】与lora微调时配置相同

#### 运行脚本

##### huggingface转megatron
```bash
bash examples/mcore/deepseek3/ckpt_convert_deepseek3_hf2mcore.sh
```

##### megatron转huggingface

```bash
bash examples/mcore/deepseek3/ckpt_convert_deepseek3_mcore2hf.sh
```

