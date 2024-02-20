#!/usr/bin/env bash

# Adapted for Librispeech clean test with OpenAI Whisper Small in float32.
# Compare https://huggingface.co/openai/whisper-small#evaluation WER 3.4%.

# Mitigate out of memory on A10 GPU (23GB).
#  --per_device_eval_batch_size 64 \

# Mitigate `ConnectionError: Server Disconnected` on Lambdalabs instance.
#  --streaming

# WER 4.06815 % is higher than without the language option.
#   --language "en" \

# Mitigate tokenizer parallelism warning message.
TOKENIZERS_PARALLELISM=false

accelerate launch run_short_form_eval.py \
  --model_name_or_path "openai/whisper-small" \
  --dataset_name "librispeech_asr" \
  --dataset_config_name "clean" \
  --dataset_split_name "test" \
  --text_column_name "text" \
  --output_dir "./" \
  --per_device_eval_batch_size 16 \
  --dtype "float32" \
  --dataloader_num_workers 16 \
  --report_to "wandb" \
  --generation_max_length 128 \
  --attn_type "flash_attn"
