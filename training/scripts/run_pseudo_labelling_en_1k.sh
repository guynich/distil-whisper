#!/usr/bin/env bash

# Small (1K examples) pseudo-labelled dataset for testing.

# Changes: mitigate out of memory problems on A10 with 22GB memory.
#  --per_device_eval_batch_size 64 \

# TODO try without `--attn_type`.

# This argument should be set for multilingual distillation only. For English speech recognition, it should be left as `None`.
# https://github.com/guynich/distil-whisper/blob/f6170f265313903513b5e5928b885974c015ce40/training/run_pseudo_labelling.py#L228
#  --language "en" \ 

accelerate launch distil-whisper/training/run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v3" \
  --dataset_name "mozilla-foundation/common_voice_13_0" \
  --dataset_config_name "en" \
  --dataset_split_name "train+validation+test" \
  --text_column_name "sentence" \
  --id_column_name "path" \
  --output_dir "./common_voice_13_0_en_pseudo_labelled_1k" \
  --wandb_project "distil-whisper-labelling" \
  --per_device_eval_batch_size 24 \
  --dtype "bfloat16" \
  --dataloader_num_workers 16 \
  --preprocessing_num_workers 16 \
  --logging_steps 500 \
  --max_label_length 128 \
  --max_samples_per_split 1000 \
  --report_to "wandb" \
  --task "transcribe" \
  --return_timestamps \
  --attn_type "flash_attn" \
  --streaming False \
  --generation_num_beams 1 \
  --decode_token_ids False
