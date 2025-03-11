CKPT_SAVE_DIR="your model save ckpt path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

python convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 2 \
       --target-pipeline-parallel-size 4 \
       --add-qkv-bias \
       --load-dir ${CKPT_LOAD_DIR} \
       --save-dir ${CKPT_SAVE_DIR} \
       --tokenizer-model ${TOKENIZER_PATH} \
       --model-type-hf llama2 \
       --params-dtype bf16