# Distil-whisper training stages.

There are four stages. Read this.
https://github.com/huggingface/distil-whisper/tree/main/training

This README documents selected information for follow the above example.
Adapted for running on A10 with 22GB RAM.

# Requirements.
Follow this.
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
python3 ./distil-whisper/training/scripts/test_working_environment.py
```
Result.
```console
Pred text:  Again, small fast craft could attack and destroy a major warship.
Environment set up successful? True
```
=> OK.

# 1. Pseudo-Labelling
https://github.com/huggingface/distil-whisper/tree/main/training#1-pseudo-labelling

Creates `~/common_voice_13_0_hi_pseudo_labelled` folder.
```console
cd
chmod +x ./distil-whisper/training/scripts/run_pseudo_labelling_hi_a10.sh
./distil-whisper/training/scripts/run_pseudo_labelling_hi_a10.sh
```

# 2. Initialization.
I used the recommended the repo name `distil-whisper-large-v2-hi`.
https://github.com/huggingface/distil-whisper/tree/main/training#2-initialisation

Script creates `distil-whisper-large-v2-hi/distil-large-v2-hi-init` student
model folder.
```console
cd
cd distil-whisper-large-v2-hi

cp ../distil-whisper/training/create_student_model.py .
cp ../distil-whisper/training/run_distillation.py .

chmod +x ~/distil-whisper/training/scripts/create_student_model_hi.sh
~/distil-whisper/training/scripts/create_student_model_hi.sh
```


