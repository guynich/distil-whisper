#!/usr/bin/env bash

# Adapted for Librispeech clean test with OpenAI Whisper Large-v2 in float32.
# Compare https://huggingface.co/openai/whisper-large-v2#evaluation WER 3.0%.

# Mitigate out of memory on A10 GPU (23GB).
#  --per_device_eval_batch_size 64 \

# Mitigate `ConnectionError: Server Disconnected` on Lambdalabs instance.
#  --streaming

# Mitigate tokenizer parallelism warning message.
TOKENIZERS_PARALLELISM=false

accelerate launch run_short_form_eval.py \
  --model_name_or_path "openai/whisper-large-v2" \
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
  --language "en" \
  --attn_type "flash_attn"
