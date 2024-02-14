# Distil-whisper training stages.

Notes for running on A10 with 22GB RAM.

There are four stages. Read this.
https://github.com/huggingface/distil-whisper/tree/main/training

# Requirements.
Follow this.  This section contains my notes.
https://github.com/huggingface/distil-whisper/tree/main/training#requirements

Clone this repo to home and add remote.
```console
git clone https://github.com/guynich/distil-whisper.git
cd distil-whisper
git remote add upstream https://github.com/huggingface/distil-whisper.git
```

Install packages.  I probably should have created a virtual environment for this
but didn't.
```console
cd training
pip install -r requirements.txt
cd ../..
```

Next run the accelerate config and select all defaults.
```console
accelerate config
```

Linked Hugging Face account.

Run the python test.
```console
cd
python3 distil_whisper/scripts/test_working_environment.py
```
Result.
```console
Pred text:  Again, small fast craft could attack and destroy a major warship.
Environment set up successful? True
```

# 1. Pseudo-Labelling
https://github.com/huggingface/distil-whisper/tree/main/training#1-pseudo-labelling

Creates `common_voice_13_0_hi_pseudo_labelled` folder under `distil-whisper`.
```console
cd
cd distil-whisper

```