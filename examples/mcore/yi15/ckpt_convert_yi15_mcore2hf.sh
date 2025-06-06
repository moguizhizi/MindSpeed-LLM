# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt.py \
	--use-mcore-models \
	--model-type-hf llama2 \
	--model-type GPT \
	--load-model-type mg \
	--save-model-type hf \
	--params-dtype bf16 \
	--target-tensor-parallel-size 1 \
	--target-pipeline-parallel-size 1 \
	--load-dir ./model_weights/Yi15-mcore \
	--save-dir ./model_from_hf/Yi15-hf/
