#!/bin/bash

set -e
set -o pipefail
model_name=${MODEL_NAME:-"mammorg"}
model_base=lmsys/vicuna-7b-v1.5
model_path=/home/jiayi/MammoRG-main/${model_name}/checkpoints/stage3

model_base="${1:-$model_base}"
model_path="${2:-$model_path}"

loader="MammoReport_test"
conv_mode="v1"
GPUS=(0) #(2 4)
CHUNKS=${#GPUS[@]}  # 自动计算GPU数量（这里是2）

for (( idx=0; idx<$CHUNKS; idx++ ))
do
    CUDA_VISIBLE_DEVICES=${GPUS[idx]} python -m llava.eval.inference \
        --loader ${loader} \
        --conv_mode ${conv_mode} \
        --temperature 0 \
        --model_path ${model_path} \
        --model_base ${model_base} \
        --chunk_idx ${idx} \
        --num_chunks ${CHUNKS} \
        --batch_size 1 \
        --group_by_length &
done
