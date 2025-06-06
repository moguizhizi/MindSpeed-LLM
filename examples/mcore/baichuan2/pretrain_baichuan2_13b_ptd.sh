#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer model path"
CKPT_LOAD_DIR="your model load ckpt path"

TP=8
PP=1


DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 40 \
    --hidden-size 5120 \
    --ffn-hidden-size 13696 \
    --num-attention-heads 40 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --seq-length 4096 \
    --disable-bias-linear \
    --max-position-embeddings 4096 \
    --micro-batch-size 2 \
    --global-batch-size 128 \
    --untie-embeddings-and-output-weights \
    --no-gradient-accumulation-fusion \
    --make-vocab-size-divisible-by 32 \
    --lr 1e-5 \
    --load ${CKPT_LOAD_DIR} \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --position-embedding-type alibi \
    --hidden-dropout 0.0 \
    --norm-epsilon 1e-6 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --use-flash-attn \
    --use-fused-swiglu \

    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --square-alibi-mask \
    --fill-neg-inf \
    --min-lr 1e-8 \
    --weight-decay 1e-4 \
    --clip-grad 1.0 \
    --seed 1234 \
    --adam-beta1 0.9 \
    --initial-loss-scale 8188.0 \
    --adam-beta2 0.98 \
    --adam-eps 1.0e-8 \
    --no-load-optim \
    --no-load-rng \
    --bf16
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/train_baichuan2_13b_mcore.log
