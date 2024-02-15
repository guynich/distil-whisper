#!/usr/bin/env bash

# Changes: mitigate out of memory problems on RTX 2080 Ti with 11GB memory.
#  --per_device_eval_batch_size 64 \
#  --dtype "bfloat16" \
#  --dataloader_num_workers 16 \

accelerate launch run_short_form_eval.py \
  --model_name_or_path "./" \
  --dataset_name "../common_voice_13_0_hi_pseudo_labelled+google/fleurs" \
  --dataset_config_name "hi+hi_in" \
  --dataset_split_name "test+test" \
  --text_column_name "sentence+transcription" \
  --output_dir "./" \
  --per_device_eval_batch_size 8 \
  --dtype "float16" \
  --dataloader_num_workers 8 \
  --report_to "wandb" \
  --generation_max_length 128 \
  --language "hi" \
  --attn_type "flash_attn" \
  --streaming
