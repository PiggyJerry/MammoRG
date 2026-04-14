#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

model_base=lmsys/vicuna-7b-v1.5
output_dir="${1:-./checkpoints}"
base_dir=${BASE_DIR:-"default_dir"}
data_path=${base_dir}/mammorg_data/split_data/Train.json

loader="MammoReport_train"

image_folder=${base_dir}/mammorg_data

model_name=${MODEL_NAME:-"default_model_name"}
vision_tower="versamammo"
vision_tower_config="${base_dir}/${model_name}/llava/model/multimodal_encoder/versamammo.json"
vision_tower_checkpoint="${base_dir}/${model_name}/llava/model/multimodal_encoder/VersaMammo.pth" 

epoch="${2:-11}"
bsz="${3:-64}"
grad_acc="${4:-1}"
lr="1e-4"
schedule="pt-${epoch}e"
run_name="stage1"
echo $run_name > run_name

deepspeed --master_port 29512 --include localhost:2 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json\
    --model_name_or_path ${model_base} \
    --version plain \
    --data_path ${data_path} \
    --loader ${loader} \
    --image_folder ${image_folder} \
    --vision_tower ${vision_tower} \
    --vision_tower_config ${vision_tower_config} \
    --vision_tower_checkpoint ${vision_tower_checkpoint} \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --freeze_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${output_dir}/${run_name} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${grad_acc} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --run_name ${run_name}\
    --report_to none \
    --aux True
    
