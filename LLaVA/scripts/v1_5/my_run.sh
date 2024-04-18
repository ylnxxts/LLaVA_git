#!/bin/bash




# Set the prompt and model versions directly in the command
CUDA_VISIBLE_DEVICES=012 deepspeed llava/train/train.py \
    --model_name_or_path /data/majunpeng/LLaVA/llava-v1.5-7b \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower /data/majunpeng/LLaVA/openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /data/majunpeng/LLaVA/llava-v1.5-7b/mm_projector.bin \
        --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --fp16 True \
    --output_dir /data/majunpeng/LLaVA/checkpoints/result \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --dataloader_num_workers 4 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb \