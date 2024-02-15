#!/usr/bin/env bash

# This config name is not found.
#  --dataset_config_name "all" \

python run_long_form_eval.py \
  --model_name_or_path "openai/whisper-large-v2" \
  --dataset_name "distil-whisper/tedlium-long-form" \
  --dataset_config_name "all" \
  --dataset_split_name "validation" \
  --text_column_name "text" \
  --output_dir "./" \
  --per_device_eval_batch_size 64 \
  --chunk_length_s 30 \
  --language "en" \
  --return_timestamps \
  --dtype "bfloat16" \
  --report_to "wandb" \
  --streaming
