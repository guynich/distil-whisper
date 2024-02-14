#!/usr/bin/env bash

# Changes.
#   --save_dir "./distil-large-v2-init"

python create_student_model.py \
  --teacher_checkpoint "openai/whisper-large-v2" \
  --encoder_layers 32 \
  --decoder_layers 2 \
  --save_dir "./distil-large-v2-hi-init"
