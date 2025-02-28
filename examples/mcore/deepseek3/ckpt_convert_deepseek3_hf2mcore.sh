# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python examples/mcore/deepseek3/convert_ckpt_deepseek3.py \
    --moe-grouped-gemm \
    --target-pipeline-parallel-size 8 \
    --target-expert-parallel-size 8 \
    --load-dir ./model_from_hf/deepseek3-bf16-hf \
    --save-dir ./model_weights/deepseek3-mcore \
    --num-layers 61 \
    --num-nextn-predict-layers 1 \
    --num-layer-list 7,7,8,8,8,8,8,7
    # --num-layer-list, --noop-layers, --num-layers-per-virtual-pipeline-stage等参数根据任务需要进行配置
