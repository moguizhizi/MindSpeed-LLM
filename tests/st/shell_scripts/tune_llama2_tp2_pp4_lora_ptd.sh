#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6016
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

CKPT_LOAD_DIR="/data/llama2-7B-tp2-pp4"
DATA_PATH="/data/tune_dataset/alpaca"
TOKENIZER_MODEL="/data/hf/llama-2-7b-hf"

TP=2
PP=4

DISTRIBUTED_ARGS=(
    --nproc_per_node $NPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

DIST_ALGO=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --sequence-parallel
)

MODEL_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 11008
    --num-attention-heads 32
    --seq-length 4096
    --max-position-embeddings 4096
)

TRAINING_ARGS=(
    --tokenizer-type PretrainedFromHF
    --tokenizer-name-or-path ${TOKENIZER_MODEL}
    --micro-batch-size 4
    --global-batch-size 32
    --make-vocab-size-divisible-by 1
    --lr 1.25e-6
    --train-iters 15
    --lr-decay-style cosine
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --attention-dropout 0.0
    --init-method-std 0.01
    --hidden-dropout 0.0
    --position-embedding-type rope
    --normalization RMSNorm
    --use-fused-rmsnorm
    --swiglu
    --use-flash-attn
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --min-lr 1.25e-7
    --weight-decay 1e-1
    --lr-warmup-fraction 0.01
    --clip-grad 1.0
    --adam-beta1 0.9
    --initial-loss-scale 65536
    --adam-beta2 0.95
    --no-gradient-accumulation-fusion
    --no-load-optim
    --no-load-rng
    --finetune
    --stage sft
    --is-instruction-dataset
    --lora-r 16
    --lora-alpha 32
    --lora-fusion
    --use-fused-mlp
    --lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h
    --variable-seq-lengths
    --recompute-num-layers 32
    --recompute-granularity full
    --recompute-method block
    --bf16
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --split 949,50,1
)

OUTPUT_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 0
)

torchrun ${DISTRIBUTED_ARGS[@]} $basepath/posttrain_gpt.py \
    ${DIST_ALGO[@]} \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    --load ${CKPT_LOAD_DIR} \
    --finetune \
    --log-throughput \
    --distributed-backend nccl
