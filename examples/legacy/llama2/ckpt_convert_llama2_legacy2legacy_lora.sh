# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --load-dir ./model_weights/llama-2-7b-hf-v0.1-tp8-pp1/ \
    --lora-load ./ckpt/llama-2-7b-lora \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_weights/llama2-7b-lora2legacy