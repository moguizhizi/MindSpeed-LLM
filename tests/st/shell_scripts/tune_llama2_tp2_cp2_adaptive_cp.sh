#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISITIC=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6015
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

CKPT_LOAD_DIR="/data/ci/Llama2-mcore-tp2/"
DATA_PATH="/data/ci/llama2_pack/alpaca"
TOKENIZER_MODEL="/data/hf/llama-2-7b-hf"

TP=2
PP=1
CP=2

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
    --context-parallel-size ${CP}
    --cp-window-size 1
    --use-fused-ring-attention-update
    --context-parallel-algo adaptive_cp_algo
    --cp-attention-mask-type general
    --adaptive-cp-manually-set-mask-list
    --adaptive-cp-dynamic-attn-mask
    --adaptive-cp-only-reschedule
    --sequence-parallel
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 11008
    --num-attention-heads 32
    --seq-length 4096
    --max-position-embeddings 4096
)

ACCELERATE_ARGS=(
    --reuse-fp32-param
    --overlap-grad-reduce
    --overlap-param-gather
    --use-distributed-optimizer
    --recompute-activation-function
    --recompute-activation-function-num-layers 1
)

TRAINING_ARGS=(
    --tokenizer-type PretrainedFromHF
    --tokenizer-name-or-path ${TOKENIZER_MODEL}
    --tokenizer-not-use-fast
    --micro-batch-size 1
    --global-batch-size 2
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
    --reset-position-ids
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
    --use-fused-swiglu
    --use-fused-rotary-pos-emb
    --overlap-grad-reduce
    --bf16
    --finetune
    --stage sft
    --is-instruction-dataset
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
    --log-throughput
)

torchrun ${DISTRIBUTED_ARGS[@]} $basepath/posttrain_gpt.py \
    ${DIST_ALGO[@]} \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${ACCELERATE_ARGS[@]} \
    --load ${CKPT_LOAD_DIR} \
    --distributed-backend nccl
