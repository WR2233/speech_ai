#!/bin/bash

# デフォルト値
METAROOT="${MODEL_NAME:-llama/hf/7B}"   #stage1
DATAROOT="${DATA_ROOT:-data/stage1}"
OUTROOT="${OUTPUT_DIR:-output/stage1}"
CACHEROOT="${DATAROOT}/cache/"
NPROC=${NPROC:-1}

mkdir -p ${CACHEROOT}/tokenized/train/
mkdir -p ${CACHEROOT}/tokenized/valid/
mkdir -p ${CACHEROOT}/group/train/
mkdir -p ${CACHEROOT}/group/valid/

#ddp related
NNODE=$1
NODE_RANK=$2
MASTER_ADDR=$3
MASTER_PORT=$4

# コマンドラインオプションで上書き
shift 4
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name-or-path)
            METAROOT="$2"
            shift 2
            ;;
        --train-file)
            TRAIN_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTROOT="$2"
            shift 2
            ;;
        --num-train-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --per-device-train-batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LR="$2"
            shift 2
            ;;
        --nproc-per-node)
            NPROC="$2"
            shift 2
            ;;
        --validation-file)
            VALIDATION_FILE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# デフォルト値設定
TRAIN_FILE="${TRAIN_FILE:-${DATAROOT}/train.txt}"
VALIDATION_FILE="${VALIDATION_FILE:-${TRAIN_FILE}}"
OUTROOT="${OUTROOT:-output/stage1}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-3}"
LR="${LR:-5e-5}"

echo "stage1: modality-adaptation pretraining"
echo "Model: $METAROOT"
echo "Train file: $TRAIN_FILE"
echo "Output dir: $OUTROOT"
echo "Epochs: $NUM_EPOCHS, Batch size: $BATCH_SIZE, LR: $LR"

torchrun \
    --nnode $NNODE \
    --nproc_per_node $NPROC \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT  \
speechgpt/src/train/ma_pretrain.py \
    --bf16 False \
    --block_size 1024 \
    --model_name_or_path "${METAROOT}" \
    --train_file ${TRAIN_FILE} \
    --validation_file ${VALIDATION_FILE} \
    --do_train \
    --output_dir "${OUTROOT}" \
    --preprocessing_num_workers 4 \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --num_train_epochs ${NUM_EPOCHS} \
    --learning_rate ${LR} \
    --log_level info \
    --logging_steps 10 \
    --save_steps 500 \
    --cache_dir ${CACHEROOT} \

