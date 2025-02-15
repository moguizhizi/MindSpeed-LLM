## DeepSeek-V3权重转换

### 使用说明

可以将已经反量化为bf16数据格式的[huggingface权重](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/tree/main)转换为mcore权重，用于微调、推理、评估等任务。反量化方法请参考DeepSeek官方提供的[代码](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/inference/fp8_cast_bf16.py)。

注：mcore权重转回huggingface权重、tp并行切分、vpp并行切分特性开发中，暂未支持。

### 启动脚本

使用DeepSeek-V3模型目录下的<a href="../../examples/mcore/deepseek3/ckpt_convert_deepseek3_hf2mcore.sh">权重转换脚本</a>。

#### 填写相关参数

【--moe-grouped-gemm】

当每个专家组有多个专家时，可以使用Grouped GEMM功能来提高利用率和性能。

【--target-pipeline-parallel-size】

流水线并行度，默认值为1。

【--target-expert-parallel-size】

专家并行度，默认值为1。

【--load-dir】

已经反量化为bf16数据格式的huggingface权重。

【--save-dir】

转换后的megatron格式权重的存储路径。

【--num-nextn-predict-layers】

MTP层的层数。如不需要MTP层，可设置为0。最大可设置为1。默认值为1。
MTP层权重默认存储在最后一个pp stage。

【--num-layers】

模型层数，该层数不包含MTP层。默认值为61。如配置空操作层，num-layers的值应为总层数（不包含MTP层）加上空操作层层数。

【--first-k-dense-replace】

moe层前的dense层数，最大可设置为3。默认值为3。

【--num-layer-list】

指定每个pp的层数，相加要等于num-layers。默认值为None。

【--noop-layers】

自定义空操作层。与--num-layer-list互斥，二者选其一使用。默认值为None。


#### 运行脚本

```bash
bash examples/mcore/deepseek3/ckpt_convert_deepseek3_hf2mcore.sh
```


