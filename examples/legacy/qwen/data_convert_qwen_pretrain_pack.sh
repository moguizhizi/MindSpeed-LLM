# Please download ShareGPT dataset, according to examples/README.md
# Please source your set_env.sh in your cann path
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

# For feature reset-position-ids, we use --append-eod to pretrain.
python ./preprocess_data.py \
    --input ./dataset/sharegpt_formatted_data-evol-gpt4.jsonl \
    --tokenizer-name-or-path ./model_from_hf/Qwen-hf/ \
    --output-prefix ./finetune_dataset/sharegpt \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name SharegptStyleInstructionHandler \
    --prompt-type qwen \
    --append-eod
    # --map-keys '{"messages":"conversations", "tags":{"role_tag": "from","content_tag": "value","user_tag": "human","assistant_tag": "gpt","system_tag": "system", "observation_tag":"observation", "function_tag":"function_call"}}' # 默认值，可不传