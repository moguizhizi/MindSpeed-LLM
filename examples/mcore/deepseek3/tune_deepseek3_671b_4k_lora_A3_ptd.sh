#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600

NPUS_PER_NODE=16
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=6000
NNODES=4
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=1
PP=4
EP=16
CP=1
CP_TYPE='ulysses_cp_algo'
NUM_LAYERS=61
SEQ_LEN=4096
MBS=1
GBS=256

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MLA_ARGS="
    --multi-head-latent-attention \
    --qk-rope-head-dim 64 \
    --qk-nope-head-dim 128 \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
"

MOE_ARGS="
    --moe-permutation-async-comm \
    --use-fused-moe-token-permute-and-unpermute \
    --moe-token-dispatcher-type alltoall \
    --first-k-dense-replace 3 \
    --moe-layer-freq 1 \
    --n-shared-experts 1 \
    --num-experts 256 \
    --moe-router-topk 8 \
    --moe-intermediate-size 2048 \
    --moe-router-load-balancing-type noaux_tc \
    --topk-group 4 \
    --routed-scaling-factor 2.5 \
    --seq-aux \
    --norm-topk-prob \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
"

ROPE_ARGS="
    --rope-scaling-beta-fast 32 \
    --rope-scaling-beta-slow 1 \
    --rope-scaling-factor 40 \
    --rope-scaling-mscale 1.0 \
    --rope-scaling-mscale-all-dim  1.0 \
    --rope-scaling-original-max-position-embeddings 4096 \
    --rope-scaling-type yarn
"

GPT_ARGS="
    --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec \
    --num-layer-list 15,15,16,15 \
    --prompt-type deepseek3 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 14 \
    --recompute-activation-function \
    --variable-seq-lengths \
    --no-shared-storage \
    --use-distributed-optimizer \
    --reuse-fp32-param \
    --use-flash-attn \
    --shape-order BNSD \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --sequence-parallel \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 7168 \
    --ffn-hidden-size 18432 \
    --num-attention-heads 128 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 163840 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-5 \
    --train-iters 1000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 1e-2 \
    --lr-warmup-iters 1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 65536 \
    --vocab-size 129280 \
    --padded-vocab-size 129280 \
    --rotary-base 10000 \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --distributed-timeout-minutes 120 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 100 \
    --eval-interval 2000 \
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng \
    --log-throughput
"


FINETUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
"


python -m torch.distributed.launch $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    $FINETUNE_ARGS \
    --save $CKPT_SAVE_DIR \
    --load $CKPT_LOAD_DIR \
    --distributed-backend nccl \
    | tee logs/tune_deepseek3_671b_4k_ptd_lora.log

