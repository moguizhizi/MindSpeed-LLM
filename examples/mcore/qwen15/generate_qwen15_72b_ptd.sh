#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1800

# please fill these path configurations
CHECKPOINT="your model directory path"
TOKENIZER_PATH="your tokenizer path"

# Change for multinode config
MASTER_ADDR=localhost
NPU_PER_NODE=8
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPU_PER_NODE*$NNODES))

TP=8
PP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPU_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS inference.py \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --ffn-hidden-size 24576 \
    --max-position-embeddings 8192 \
    --seq-length 8192 \
    --padded-vocab-size 152064 \
    --rotary-base 1000000 \
    --make-vocab-size-divisible-by 1 \
    --attention-softmax-in-fp32 \
    --no-load-optim \
    --no-load-rng \
    --untie-embeddings-and-output-weights \
    --micro-batch-size 1 \
    --swiglu \
    --disable-bias-linear \
    --add-qkv-bias \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --load ${CHECKPOINT} \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --position-embedding-type rope \
    --tokenizer-not-use-fast \
    --hidden-dropout 0 \
    --attention-dropout 0 \
    --max-new-tokens 256 \
    --bf16 \
    --seed 42 \
    | tee logs/generate_mcore_qwen15_72b.log