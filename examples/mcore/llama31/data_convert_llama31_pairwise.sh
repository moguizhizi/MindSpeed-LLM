# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p ./pairwise_dataset

python ./preprocess_data.py \
        --input ./dataset/orca_rlhf.jsonl \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-not-use-fast \
        --tokenizer-name-or-path ./model_from_hf/llama31-hf/ \
        --output-prefix ./pairwise_dataset/orca_rlhf_llama31 \
        --workers 4 \
        --log-interval 1000 \
        --handler-name AlpacaStylePairwiseHandler \
        --prompt-type llama3 \
        --seq-length 8192 \
        --map-keys '{"prompt":"question", "query":"", "system":"system"}'
