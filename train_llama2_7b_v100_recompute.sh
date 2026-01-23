#!/bin/bash

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
#export LOG_LEVEL=${LOG_LEVEL:-INFO}
#export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-19}
#export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
#export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
#export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
#export NCCL_AVOID_RECORD_STREAMS=${NCCL_AVOID_RECORD_STREAMS:-1}

# CHECKPOINT_PATH=${1:-"checkpoints/llama2_7b_fp8"}
TENSORBOARD_LOGS_PATH="tensorboard_logs/llama2_7b"
TOKENIZER_ARG="/data/home/menglin/Llama-2-7b/tokenizer.model" # Path to tokenizer model, or "MOCK"
DATA_ARG="/data/home/menglin/RedPajama-Data-1T-Sample/preprocess/preprocess_text_document"     # Data prefix, or "MOCK"

# Create directories if they don't exist
# mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# Distributed training setup

NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}

# Path to the pretrain_gpt.py script, assuming this script is run from the root of the Megatron-LM repository
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Fixed model and training parameters
TP_SIZE=${TP_SIZE:-1}   # Tensor parallelism size
CP_SIZE=${CP_SIZE:-1}   # Context parallelism size
PP_SIZE=${PP_SIZE:-1}   # Pipeline parallelism size
DP_SIZE=${DP_SIZE:-1}   # Data parallelism size
GPUS_PER_NODE=$((TP_SIZE * DP_SIZE * PP_SIZE))
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2} 
GLOBAL_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}  # Reduced global batch size to fit V100 memory
NUM_LAYERS=${NUM_LAYERS:-2} # Reduced number of layers to fit V100 memory
DTYPE=${DTYPE:-fp16}
SEQ_LENGTH=${SEQ_LENGTH:-1024}  # Reduced sequence length to fit V100 memory
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-2048}  # Reduced max position embeddings to fit V100 memory
HIDDEN_SIZE=${HIDDEN_SIZE:-4096}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-11008}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-1}
NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS:-1}

RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-'full'}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-'uniform'}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-2}


# Data cache path (useful for both mock and real data)
DATA_CACHE_PATH="${PWD}/benchmark_cache_llama2_7b_fp16"
mkdir -p "$DATA_CACHE_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE  # Adjusted hidden size for 7B model
    --ffn-hidden-size $FFN_HIDDEN_SIZE  # Adjusted FFN hidden size for 7B model
    --num-attention-heads $NUM_ATTENTION_HEADS
    --group-query-attention
    --num-query-groups $NUM_QUERY_GROUPS
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 1000000 
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.0134
    --attention-backend fused
    --apply-layernorm-1p 
    --untie-embeddings-and-output-weights
    --disable-bias-linear 
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 1000000
    --lr-decay-samples 990000
    --lr-warmup-samples 10000
    --lr 0.00015
    --min-lr 0.00001
    --no-rope-fusion
    # --decoupled-lr 5.0e-4      # Specific to decoupled AdamW, ensure optimizer is compatible
    # --decoupled-min-lr 4.5e-5  # Specific to decoupled AdamW
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --fp16
    # --grad-reduce-in-bf16
    # --cross-entropy-loss-fusion
    --calculate-per-token-loss 
    --manual-gc 
    --empty-unused-memory-level 1 
    --exit-interval 20
    
)


# Conditional arguments based on DTYPE (FP8)
DTYPE_ARGS=()
if [[ "$DTYPE" == "fp8" ]]; then
    DTYPE_ARGS+=(
        "--fp8-format hybrid"
        "--fp8-amax-history-len 1024"
        "--fp8-amax-compute-algo max"
        "--fp8-param-gather"
    )
fi

RECOMPUTE_ARGS=()

case "$RECOMPUTE_GRANULARITY" in
  null)
    RECOMPUTE_ARGS=()
    ;;
  selective)
    RECOMPUTE_ARGS=(
      --recompute-granularity "$RECOMPUTE_GRANULARITY"
    )
    ;;
  full)
    RECOMPUTE_ARGS=(
      --recompute-granularity "$RECOMPUTE_GRANULARITY"
      --recompute-method "$RECOMPUTE_METHOD"
      --recompute-num-layers "$RECOMPUTE_NUM_LAYERS"
    )
    ;;
  *)
    echo "[ERROR] Unknown RECOMPUTE_GRANULARITY: $RECOMPUTE_GRANULARITY" >&2
    exit 1
    ;;
esac

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    --pipeline-model-parallel-size $PP_SIZE # Not explicitly set in llama script options, assume 1 if not multi-node PP
    --sequence-parallel  # Always enable sequence parallelism with TP_SIZE=2
)

# Distributed Data Parallel (DDP) arguments
# From original script's ddp_args
DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
TRAINING_ARGS+=("${DDP_ARGS[@]}")

# Data arguments (conditional for mock vs real data)
DATA_ARGS_LIST=()
if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]] || [[ -z "$TOKENIZER_ARG" ]]; then
    DATA_ARGS_LIST+=(
        "--mock-data"
        "--tokenizer-type NullTokenizer"
        "--vocab-size 128256" 
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--tiktoken-pattern v2" 
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
    )
else
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--data-path $DATA_ARG"
        "--tokenizer-type SentencePieceTokenizer" 
        "--tokenizer-model $TOKENIZER_ARG"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
        # Note: --vocab-size might be inferred by HuggingFaceTokenizer or might need to be explicit.
        "--vocab-size 32000"
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 200
    --eval-interval 200
    # --save-interval 1000
    --log-throughput
    # --profile
    # --use-pytorch-profiler
    # --profile-step-start 4
    # --profile-step-end 5
    # --profile-ranks 0              # 只采 rank0，减少文件体积
    --ckpt-format torch_dist 
    --distributed-timeout-minutes 60
    --log-timers-to-tensorboard
    # --save "$CHECKPOINT_PATH"
    # --load "$CHECKPOINT_PATH" 
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
)

# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and pretrain_gpt.py is present."
    exit 1
fi

echo "[CMD] torchrun ${DISTRIBUTED_ARGS[@]} \"${PRETRAIN_SCRIPT_PATH}\" \\"
printf "      %s \\\n" "${MODEL_ARGS[@]}" "${TRAINING_ARGS[@]}" "${RECOMPUTE_ARGS[@]}" \
        "${DTYPE_ARGS[@]}" "${MODEL_PARALLEL_ARGS[@]}" "${DATA_ARGS_LIST[@]}" "${EVAL_AND_LOGGING_ARGS[@]}"

# Run the training command
torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${RECOMPUTE_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

set +x