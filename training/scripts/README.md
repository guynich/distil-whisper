# Distil-whisper training stages.

There are four stages. Read this.
https://github.com/huggingface/distil-whisper/tree/main/training

This README documents selected information for follow the above example.
Adapted for running on A10 GPU with 22GB RAM. I added scripts for a
workstation with RTX 2080 Ti GPU (11GB RAM).

- [Distil-whisper training stages.](#distil-whisper-training-stages)
- [Requirements.](#requirements)
- [1. Pseudo-Labelling](#1-pseudo-labelling)
  - [Different workstation with RTX 2080 Ti GPU.](#different-workstation-with-rtx-2080-ti-gpu)
- [2. Initialization.](#2-initialization)
- [3. Training.](#3-training)
  - [Training error (on Ubuntu with A10 GPU).](#training-error-on-ubuntu-with-a10-gpu)
    - [Check dependencies.](#check-dependencies)
  - [Training on workstation with RTX 2080 Ti GPU.](#training-on-workstation-with-rtx-2080-ti-gpu)
- [4. Evaluation.](#4-evaluation)
  - [Short Form on RTX 2080 Ti.](#short-form-on-rtx-2080-ti)

# Requirements.

Update/upgrade your environment.
```console
sudo apt install
sudo apt upgrade -y

pip install --upgrade pip
```

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

## Different workstation with RTX 2080 Ti GPU.

I provide a script for a different workstation with RTX 2080 Ti.
```console
cd
chmod +x ./distil-whisper/training/scripts/run_pseudo_labelling_hi_rtx2080ti.sh
./distil-whisper/training/scripts/run_pseudo_labelling_hi_rtx2080ti.sh
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

## Training error (on Ubuntu with A10 GPU).

```console
cd
cd distil-whisper-large-v2-hi

chmod +x ~/distil-whisper/training/scripts/run_distillation_hi_a10.sh
~/distil-whisper/training/scripts/run_distillation_hi_a10.sh
```

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

### Check dependencies.
I ran another install on a workstation with NVidia RTX 2080 Ti GPU and the
workflow through Stage 3 training runs without `BuilderConfig` error.

First I checked the environment on instance with A10 GPU.
```console
sudo apt install
sudo apt upgrade -y

sudo reboot

pip install --upgrade pip

cd distil-whisper/training
pip install --upgrade -r requirements.txt
cd
```

Second I inspected versions for `datasets` 2.17.0, `transformers` 4.37.2,
`torch` 2.2.0, `evaluate` 0.4.1 are same on both the RTX 2080 Ti workstation and
on the instance with A10 GPU.  But the `accelerate`  package version on
workstation was 0.27.0 not 0.27.2 on instance with A10 GPU.

I downgrading `accelerate` on the instance with A10 GPU.
```console
pip install --force-reinstall accelerate==0.27.0

# Requires downgrading the folllowing.
pip install --force-reinstall fsspec==2023.10.0
pip install --force-reinstall numpy==1.23
```

Initially this did not mitigate the error triggered by training script.
`ValueError: BuilderConfig 'hi' not found. Available: ['default']`
But then the error is not seen after rerunning the pseudo-labelling
[script](#1-pseudo-labelling).

=> Training script now runs.

## Training on workstation with RTX 2080 Ti GPU.

It has less memory than the A10 GPU in last section.  But the training script
proceeds without `BuilderConfig` error seen above on different instance.

```console
cd
cd distil-whisper-large-v2-hi

chmod +x ~/distil-whisper/training/scripts/run_distillation_hi_rtx2080ti.sh
~/distil-whisper/training/scripts/run_distillation_hi_rtx2080ti.sh
```

Result.
```console
wandb: Run summary:
wandb:        eval/ce_loss 1.4156
wandb:          eval/epoch 2
wandb:        eval/kl_loss 1.28667
wandb:           eval/loss 2.41915
wandb:           eval/time 1470.33953
wandb:            eval/wer 41.70491
wandb:      eval/wer_ortho 59.77059
wandb:       train/ce_loss 0.49456
wandb:         train/epoch 2
wandb:       train/kl_loss 0.33438
wandb: train/learning_rate 0.0001
wandb:          train/loss 0.73003
wandb:          train/time 1145.25749
```
Metric `eval/wer` is `41.7%`.  The `distil-whisper` README
[here](https://github.com/huggingface/distil-whisper/tree/main/training#3-training)
saw a "final WER of 31%" for their script values.

# 4. Evaluation.

https://github.com/huggingface/distil-whisper/blob/main/training/README.md#4-evaluation

## Short Form on RTX 2080 Ti.

```console
cd
cd distil-whisper-large-v2-hi

cp ../distil-whisper/training/run_short_form_eval.py .

chmod +x ~/distil-whisper/training/scripts/run_short_form_eval_hi_rtx2080ti.sh
~/distil-whisper/training/scripts/run_short_form_eval_hi_rtx2080ti.sh
```

Triggers following warning.
```console
02/15/2024 09:07:41 - WARNING - datasets.iterable_dataset - Too many dataloader workers: 8 (max is dataset.n_shards=1). Stopping 7 dataloader workers.
```

Here are the short form evaluation results.
```console
wandb: Run history:
wandb:      test/time █▁
wandb:       test/wer ▁█
wandb: test/wer_ortho ▁█
wandb:
wandb: Run summary:
wandb:      test/time 70.2072
wandb:       test/wer 66.55725
wandb: test/wer_ortho 82.77084
```
Metric `test/wer` is `66.6%`.
