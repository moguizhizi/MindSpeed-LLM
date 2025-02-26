# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./grpo_dataset

python ./preprocess_data.py \
--input ../pe-nlp/train-00000-of-00001.parquet \
--tokenizer-name-or-path ../Qwen2.5-7B \
--output-prefix ./grpo_dataset/pe_nlp \
--handler-name R1AlpacaStyleInstructionHandler \
--tokenizer-type PretrainedFromHF \
--workers 4 \
--log-interval 1000 \
--map-keys '{"prompt":"question", "query":"", "response":"ground_truth_answer"}'  \
--dataset-additional-keys labels \
--prompt-type qwen_r1