# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
--input datasets/firefly-train-20k_cot_formatted.parquet \
--tokenizer-name-or-path ../Qwen2.5-7B \
--output-prefix ./finetune_dataset/firefly \
--handler-name AlpacaStyleInstructionHandler \
--tokenizer-type PretrainedFromHF \
--workers 4 \
--log-interval 1000 \
--map-keys '{"prompt":"problem", "query":"", "response":"reannotated_assistant_content"}'  \
--prompt-type qwen