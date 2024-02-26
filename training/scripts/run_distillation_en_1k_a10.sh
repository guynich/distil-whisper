#!/usr/bin/env bash

# Stage 3.
# https://github.com/huggingface/distil-whisper/tree/main/training#3-training

# Changes for stdudent model path and A10 GPU with 22GB memory.
#  --model_name_or_path "./distil-large-v2-init" \
#  --per_device_train_batch_size 64 \
#  --per_device_eval_batch_size 64 \

# TODO try batch_size of 16.
accelerate launch run_distillation.py \
  --model_name_or_path "./distil-large-v2-hi-init" \
  --teacher_model_name_or_path "openai/whisper-large-v2" \
  --train_dataset_name "../common_voice_13_0_en_pseudo_labelled_1k+../common_voice_13_0_en_pseudo_labelled_1k" \
  --train_dataset_config_name "default+default" \
  --train_split_name "train+validation" \
  --text_column_name "sentence+sentence" \
  --train_dataset_samples "10+5" \
  --eval_dataset_name "../common_voice_13_0_en_pseudo_labelled_1k" \
  --eval_dataset_config_name "default" \
  --eval_split_name "test" \
  --eval_text_column_name "sentence" \
  --eval_steps 1000 \
  --save_steps 1000 \
  --warmup_steps 50 \
  --learning_rate 0.0001 \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --save_total_limit 1 \
  --max_steps 5000 \
  --wer_threshold 10 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --dataloader_num_workers 16 \
  --preprocessing_num_workers 16 \
  --ddp_timeout 7200 \
  --dtype "bfloat16" \
  --output_dir "./" \
  --do_train \
  --do_eval \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --predict_with_generate \
  --freeze_encoder \
  --streaming False \
  --push_to_hub
