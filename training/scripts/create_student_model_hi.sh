#!/usr/bin/env bash

# Stage 2.
# https://github.com/huggingface/distil-whisper/tree/main/training#2-initialisation

# Run from new repo folder with copied scripts - see link above.
#   cd
#   cd distil-whisper-large-v2-hi
#   ~/distil-whisper

# Changes.
#   --save_dir "./distil-large-v2-init"

python create_student_model.py \
  --teacher_checkpoint "openai/whisper-large-v2" \
  --encoder_layers 32 \
  --decoder_layers 2 \
  --save_dir "./distil-large-v2-hi-init"
