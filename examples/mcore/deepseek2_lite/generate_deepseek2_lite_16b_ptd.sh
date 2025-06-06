#!/bin/bash

# The number of parameters is not aligned
export HCCL_CONNECT_TIMEOUT=1200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# please fill these path configurations
CHECKPOINT="Your ckpt file path"
TOKENIZER_PATH="Your vocab file path"
TOKENIZER_MODEL="Your vocab model file path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=1
EP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

python -m torch.distributed.launch $DISTRIBUTED_ARGS inference.py \
    --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec \
    --load "${CHECKPOINT}" \
    --task chat \
    --max-new-tokens 256 \
    --use-mcore-models \
    --moe-grouped-gemm \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 27 \
    --hidden-size 2048 \
    --ffn-hidden-size 10944 \
    --num-attention-heads 16 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path "${TOKENIZER_MODEL}" \
    --seq-length 8192 \
    --max-position-embeddings 163840 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --make-vocab-size-divisible-by 1 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-rotary-position-embeddings \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --vocab-size 102400 \
    --padded-vocab-size 102400 \
    --rotary-base 10000 \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --multi-head-latent-attention \
    --qk-rope-head-dim 64 \
    --qk-nope-head-dim 128 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
    --expert-model-parallel-size ${EP} \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type allgather \
    --first-k-dense-replace 1 \
    --moe-layer-freq 1 \
    --n-shared-experts 2 \
    --num-experts 64 \
    --moe-router-topk 6 \
    --moe-intermediate-size 1408 \
    --moe-router-load-balancing-type softmax_topk \
    --topk-group 1 \
    --moe-aux-loss-coeff 0.001 \
    --routed-scaling-factor 1.0 \
    --seq-aux \
    --rope-scaling-beta-fast 32 \
    --rope-scaling-beta-slow 1 \
    --rope-scaling-factor  40 \
    --rope-scaling-mscale 0.707 \
    --rope-scaling-mscale-all-dim  0.707 \
    --rope-scaling-original-max-position-embeddings 4096 \
    --rope-scaling-type yarn \
    --distributed-backend nccl \
    | tee logs/generate_deepseek2_lite.log

