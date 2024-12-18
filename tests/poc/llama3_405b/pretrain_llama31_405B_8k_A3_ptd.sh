#!/bin/bash

export HCCL_DETERMINISTIC=true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=2400

export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=2

###############指定训练脚本执行路径###############
rm -rf /root/.cache
NPUS_PER_NODE=16
MASTER_ADDR="master node ip" #主节点IP
MASTER_PORT=6000
NNODES=16
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=16
PP=8
VPP=1
CP=2
CP_TYPE='megatron_cp_algo'
NUM_LAYERS=128
SEQ_LEN=8192
MBS=1
GBS=128

DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer model path"

time=$(date +%Y%m%d%H%M)


PROF_ARGS="
    --profile \
    --profile-step-start 5 \
    --profile-step-end 6 \
    --profile-level level1 \
    --profile-with-cpu \
    --profile-with-memory \
    --profile-save-path ./profile_dir \
"


GPT_ARGS="
    --use-mcore-models \
    --reuse-fp32-param \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage ${VPP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo megatron_cp_algo \
    --use-cp-send-recv-overlap \
    --log-throughput \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-distributed-optimizer \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 16384 \
    --ffn-hidden-size 53248 \
    --num-attention-heads 128 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size 1 \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-6 \
    --train-iters 5000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --use-fused-rotary-pos-emb \
    --use-fused-swiglu \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 4096.0 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --no-gradient-accumulation-fusion \
    --disable-bias-linear \
    --group-query-attention \
    --num-query-groups 16 \
    --lr-warmup-fraction 0.01 \
    --bf16 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1 \
    --no-shared-storage
"

RECOMPUTE_ARGS="
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 9 \
    --enable-recompute-layers-per-pp-rank \
    --recompute-in-advance \
"

ACT_RECOMPUTE_ARGS="
   --swap-attention \
   --recompute-activation-function \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

TP_2D_ARGS="
    --tp-2d \
    --tp-x 8 \
    --tp-y 2 \
"

NO_TP_2D_ARGS="
    --sequence-parallel \
    --use-fused-rmsnorm \
"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $TP_2D_ARGS \
    --distributed-backend nccl
