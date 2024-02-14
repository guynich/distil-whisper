# Distil-whisper training stages.

There are four stages. Read this.
https://github.com/huggingface/distil-whisper/tree/main/training

This README documents selected information for follow the above example.
Adapted for running on A10 with 22GB RAM.

- [Distil-whisper training stages.](#distil-whisper-training-stages)
- [Requirements.](#requirements)
- [1. Pseudo-Labelling](#1-pseudo-labelling)
- [2. Initialization.](#2-initialization)
- [3. Training.](#3-training)
  - [Training error.](#training-error)

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

I saw https://github.com/huggingface/distil-whisper/pull/76 and
made the same change in this repo
[code](https://github.com/guynich/distil-whisper/blob/1daaeecae324801440193b8fb57c0791e5ecdc7e/training/run_pseudo_labelling.py#L535).

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

# 3. Training.
https://github.com/huggingface/distil-whisper/tree/main/training#3-training

```console
cd
cd distil-whisper-large-v2-hi

chmod +x ~/distil-whisper/training/scripts/run_distillation_hi_a10.sh
~/distil-whisper/training/scripts/run_distillation_hi_a10.sh
```

## Training error.

This `run_distillation_hi_a10.sh` script triggers the following ValueError.
```console
Traceback (most recent call last):
  File "/home/ubuntu/distil-whisper-large-v2-hi/run_distillation.py", line 1668, in <module>
    main()
  File "/home/ubuntu/distil-whisper-large-v2-hi/run_distillation.py", line 829, in main
    raw_datasets["train"] = load_multiple_datasets(
  File "/home/ubuntu/distil-whisper-large-v2-hi/run_distillation.py", line 591, in load_multiple_datasets
    dataset = load_dataset(
  File "/home/ubuntu/.local/lib/python3.10/site-packages/datasets/load.py", line 2548, in load_dataset
    builder_instance = load_dataset_builder(
  File "/home/ubuntu/.local/lib/python3.10/site-packages/datasets/load.py", line 2257, in load_dataset_builder
    builder_instance: DatasetBuilder = builder_cls(
  File "/home/ubuntu/.local/lib/python3.10/site-packages/datasets/builder.py", line 371, in __init__
    self.config, self.config_id = self._create_builder_config(
  File "/home/ubuntu/.local/lib/python3.10/site-packages/datasets/builder.py", line 592, in _create_builder_config
    raise ValueError(
ValueError: BuilderConfig 'hi' not found. Available: ['default']
```

The pseudo-labelling [script](/training/scripts/run_pseudo_labelling_hi_a10.sh)
sets dataset_config_name to "hi".
