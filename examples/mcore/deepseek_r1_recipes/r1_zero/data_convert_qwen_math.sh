# 请根据 examples/README.md 下 “数据集准备及处理” 章节下载 Alpaca 数据集
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
        --input ./dataset/pe-nlp/train-00000-of-00001.parquet \
        --tokenizer-name-or-path ./models/Qwen2.5-Math-7B \
        --output-prefix ./dataset/pe-nlp/data \
        --handler-name R1AlpacaStyleInstructionHandler \
        --tokenizer-type PretrainedFromHF \
        --workers 4 \
        --log-interval 1000 \
        --prompt-type qwen_math_r1 \
        --dataset-additional-keys labels \
        --map-keys '{"prompt":"question", "query":"", "response": "ground_truth_answer", "system":""}' \
