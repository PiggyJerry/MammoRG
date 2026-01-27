#!/bin/bash

set -e
set -o pipefail
model_name=${MODEL_NAME:-"mammorg"}
base_dir=${BASE_DIR:-"default_dir"}
model_base=lmsys/vicuna-7b-v1.5
model_path=${base_dir}/${model_name}/checkpoints/stage3

model_base="${1:-$model_base}"
model_path="${2:-$model_path}"
prediction_dir="${3:-results/${model_name}}"
prediction_file=$prediction_dir/vindr_mammo

run_name="${4:-${model_name}}"


query_file=${base_dir}/mammorg_data/split_data/vindr_mammo.json

image_folder=${base_dir}/mammorg_data
loader="MammoReport_test"
conv_mode="v1"
GPUS=(0) #(2 4)
CHUNKS=${#GPUS[@]} 

for (( idx=0; idx<$CHUNKS; idx++ ))
do
    CUDA_VISIBLE_DEVICES=${GPUS[idx]} python -m llava.eval.model_eval_clslabel \
        --query_file ${query_file} \
        --loader ${loader} \
        --image_folder ${image_folder} \
        --conv_mode ${conv_mode} \
        --prediction_file ${prediction_file}_${idx}.jsonl \
        --temperature 0 \
        --model_path ${model_path} \
        --model_base ${model_base} \
        --chunk_idx ${idx} \
        --num_chunks ${CHUNKS} \
        --batch_size 8 \
        --group_by_length &
done