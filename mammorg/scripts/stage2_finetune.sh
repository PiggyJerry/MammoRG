#!/bin/bash

# Set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1

model_base=lmsys/vicuna-7b-v1.5
output_dir="${1:-./checkpoints}"
base_dir=${BASE_DIR:-"default_dir"}
model_name=${MODEL_NAME:-"default_model_name"}
PRETRAIN="${base_dir}/${model_name}/checkpoints/stage2_alignment/non_lora_trainables.bin"

vision_tower="versamammo"
vision_tower_config="${base_dir}/${model_name}/llava/model/multimodal_encoder/versamammo.json"
vision_tower_checkpoint="${base_dir}/${model_name}/llava/model/multimodal_encoder/VersaMammo.pth" 

data_path=${base_dir}/mammorg_data/split_data/Train_small.json
loader="MammoReport_train"
image_folder=${base_dir}/mammorg_data

epoch="${2:-3}"
bsz="${3:-4}"
grad_acc="${4:-1}"
lr="1e-4"
schedule="lora-${epoch}e"
export run_name="stage2_finetune"
echo $run_name > run_name

deepspeed --master_port 29512 --include localhost:2 ${base_dir}/${model_name}/llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_alpha 128 \
    --model_name_or_path ${model_base} \
    --version $PROMPT_VERSION \
    --data_path ${data_path} \
    --loader ${loader} \
    --image_folder ${image_folder} \
    --vision_tower ${vision_tower} \
    --vision_tower_config ${vision_tower_config} \
    --vision_tower_checkpoint ${vision_tower_checkpoint} \
    --pretrain ${PRETRAIN} \
    --mm_projector_type mlp2x_gelu \
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
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to none \
    --run_name ${run_name}
