# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --save-dir ./model_from_hf/qwen3_32b_hf/ \
    --load-dir ./model_weights/qwen3_32b_mcore/ \
    --model-type-hf qwen3 \
    --params-dtype bf16
