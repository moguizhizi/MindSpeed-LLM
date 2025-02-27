#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

POLICY_MODEL_PATH=/home/cq/code/MindSpeed-LLM/model_from_hf/Qwen2.5-7B-Instruct
DATA_PATH=generate_output.jsonl

python mindspeed_llm/tasks/posttrain/rejection_sampling/rejection_sampling.py \
   --pretrain $POLICY_MODEL_PATH \
   --task rejection_sampling \
   --map-keys '{"prompt":"prompt", "gt_answer":"gt_answer", "response":"output"}' \
   --use-ground-truth-answer \
   --max-new-tokens 2048 \
   --prompt-max-len 2048 \
   --dataset $DATA_PATH \
   --temperature 0.3 \
   --top-p 0.3 \
   --repetition-penalty 1.05 \
   --enable-prefix-caching \
   --tp-size 4 \
   --output-path rejection_sampling_output.jsonl
