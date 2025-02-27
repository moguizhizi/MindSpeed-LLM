# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
	--input rejection_sampling_output.jsonl \
	--tokenizer-name-or-path ./model_from_hf/qwen2.5_14b_hf/ \
	--output-prefix ./finetune_dataset/rejection_sampling_output \
  --handler-name AlpacaStyleInstructionHandler \
	--tokenizer-type PretrainedFromHF \
	--workers 4 \
	--log-interval 1000 \
  --prompt-type qwen \
  --map-keys '{"prompt":"prompt","query":"","response":"output"}'