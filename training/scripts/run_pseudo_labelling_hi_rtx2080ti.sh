#!/usr/bin/env bash

# Changes: mitigate out of memory problems on RTX 2080 Ti with 11GB memory.
#  --per_device_eval_batch_size 64 \
#  --dtype "bfloat16"
#  --dataloader_num_workers 16 \
#  --preprocessing_num_workers 16 \

accelerate launch distil-whisper/training/run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v2" \
  --dataset_name "mozilla-foundation/common_voice_13_0" \
  --dataset_config_name "hi" \
  --dataset_split_name "train+validation+test" \
  --text_column_name "sentence" \
  --id_column_name "path" \
  --output_dir "./common_voice_13_0_hi_pseudo_labelled" \
  --wandb_project "distil-whisper-labelling" \
  --per_device_eval_batch_size 8 \
  --dtype "float16" \
  --dataloader_num_workers 8 \
  --preprocessing_num_workers 1 \
  --logging_steps 500 \
  --max_label_length 128 \
  --report_to "wandb" \
  --language "hi" \
  --task "transcribe" \
  --return_timestamps \
  --attn_type "flash_attn" \
  --streaming False \
  --generation_num_beams 1 \
  --decode_token_ids False \
  --push_to_hub
