#!/bin/bash

set -e
set -o pipefail
model_name=${MODEL_NAME:-"default_model_name"}
base_dir=${BASE_DIR:-"default_dir"}
model_base=lmsys/vicuna-7b-v1.5
model_path=${base_dir}/${model_name}/checkpoints/stage3

model_base="${1:-$model_base}"
model_path="${2:-$model_path}"
prediction_dir="${3:-results/${model_name}}"
prediction_file=$prediction_dir/Test

run_name="${4:-${model_name}}"
query_file=${base_dir}/mammorg_data/split_data/Test.json

image_folder=${base_dir}/mammorg_data
loader="MammoReport_test"
conv_mode="v1"
GPUS=(1) #(2 4)
CHUNKS=${#GPUS[@]} 

for (( idx=0; idx<$CHUNKS; idx++ ))
do
    CUDA_VISIBLE_DEVICES=${GPUS[idx]} python -m llava.eval.model_eval \
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

wait

cat ${prediction_file}_*.jsonl > mammorg_preds.jsonl

pushd ${base_dir}/${model_name}/llava/eval/rrg_eval
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") python run.py ../../../mammorg_preds.jsonl --run_name ${run_name} --output_dir ../../../${prediction_dir}/eval
popd

rm mammorg_preds.jsonl