# Distil-whisper training stages.

There are four stages. Read this.
https://github.com/huggingface/distil-whisper/tree/main/training

This README documents selected information for follow the above example.
Adapted for running on A10 GPU with 23GB RAM. I added scripts for a
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
  - [Short Form on A10 GPU.](#short-form-on-a10-gpu)
  - [Short Form on RTX 2080 Ti GPU.](#short-form-on-rtx-2080-ti-gpu)
  - [Long Form error on A10 GPU.](#long-form-error-on-a10-gpu)
- [Other.](#other)
  - [Librispeech clean test English.](#librispeech-clean-test-english)
    - [Open AI Large-v2 model.](#open-ai-large-v2-model)
    - [Open AI Small model.](#open-ai-small-model)
    - [Open AI Tiny model.](#open-ai-tiny-model)
    - [Summary.](#summary)

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

```console
wandb: Run summary:
wandb:        eval/ce_loss 1.27944
wandb:          eval/epoch 18
wandb:        eval/kl_loss 1.11462
wandb:           eval/loss 2.13817
wandb:           eval/time 1114.91925
wandb:            eval/wer 28.6298
wandb:      eval/wer_ortho 46.51952
wandb:       train/ce_loss 0.09619
wandb:         train/epoch 18
wandb:       train/kl_loss 0.19989
wandb: train/learning_rate 0.0001
wandb:          train/loss 0.27684
wandb:          train/time 3372.43
```
Metric `eval/wer` is `28.6%`.  The `distil-whisper` README
[here](https://github.com/huggingface/distil-whisper/tree/main/training#3-training)
saw a "final WER of 31%" for their script values.

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
Metric `eval/wer` is `41.7%` usingf float16.  The `distil-whisper` README
[here](https://github.com/huggingface/distil-whisper/tree/main/training#3-training)
saw a "final WER of 31%" for their script values.

# 4. Evaluation.

https://github.com/huggingface/distil-whisper/blob/main/training/README.md#4-evaluation

## Short Form on A10 GPU.

```console
cd
cd distil-whisper-large-v2-hi

cp ../distil-whisper/training/run_short_form_eval.py .

chmod +x ~/distil-whisper/training/scripts/run_short_form_eval_hi_a10.sh
~/distil-whisper/training/scripts/run_short_form_eval_hi_a10.sh
```
Triggered this warning.
```console
Evaluating common_voice_13_0_hi_pseudo_labelled/test...: 0it [00:00, ?it/s]Too many dataloader workers: 16 (max is dataset.n_shards=1). Stopping 15 dataloader workers.
02/15/2024 21:49:37 - WARNING - datasets.iterable_dataset - Too many dataloader workers: 16 (max is dataset.n_shards=1). Stopping 15 dataloader workers.
```

```console
wandb: Run history:
wandb:      test/time █▁
wandb:       test/wer ▁█
wandb: test/wer_ortho ▁█
wandb:
wandb: Run summary:
wandb:      test/time 37.77144
wandb:       test/wer 56.91455
wandb: test/wer_ortho 71.35688
```
Metric `test/wer` on out of distribution (OOD) FLEURS test set is `56.9%` for student model trained in `bfloat16` precision.

Note the console prints two WER values: the first `27.0` is for the in distribution (ID) pseudo-labelled common voice 13.0 dataset, the second above is for the OOD FLEURS test set.  The wandb [test/wer plot](https://wandb.ai/guynich/distil-whisper/runs/g0e52ufc?workspace=user-guynich) shows these two values as discrete points versus step.

## Short Form on RTX 2080 Ti GPU.

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
Metric `test/wer` out of distribution (OOD) FLEURS test set is `66.6%` for student model trained in `float16` precision.

Note the console prints two WER values: the first is for the in distribution (ID) pseudo-labelled common voice 13.0 dataset, the second above is for the OOD FLEURS test set.

## Long Form error on A10 GPU.
https://github.com/huggingface/distil-whisper/blob/main/training/README.md#long-form
```
The script run_long_form_eval.py can be used to evaluate the trained student model on an arbitrary number of long-form evaluation sets. Since we don't have a long-form validation set for Hindi to hand, we'll evaluate the teacher model on the TED-LIUM validation set in this English example.
```

```console
cd
cd distil-whisper-large-v2-hi

cp ../distil-whisper/training/run_long_form_eval.py .

chmod +x ~/distil-whisper/training/scripts/run_long_form_eval_en_a10.sh
~/distil-whisper/training/scripts/run_long_form_eval_en_a10.sh
```

The given bash script `--dataset_config_name "all"` triggers an error.
```console
ValueError: BuilderConfig 'all' not found. Available: ['default']
```

# Other.

Miscellaneous.

## Librispeech clean test English.

### Open AI Large-v2 model.

Large-v2 model card https://huggingface.co/openai/whisper-large-v2#evaluation
WER 3.000%.

Evaluate Whisper model on A10 GPU.
* float32 precision.
* batch size set to 16 to mitigate out of memory.
* option `--language` is not set (if set `--language "en"` WER is 3.1683 %).
* option `-streaming` is not set (if set `--streaming` server disconnection seen after 35 iterations).
* environment variable TOKENIZERS_PARALLELISM=false to mitigate warning message.
```console
cd
cd distil-whisper/training

chmod +x scripts/run_short_form_eval_en_librispeech_large_v2.sh

tmux  # Optional.

./scripts/run_short_form_eval_en_librispeech_large_v2.sh
```

Warning `02/16/2024 18:45:46 - WARNING - datasets.iterable_dataset - Too many dataloader workers: 16 (max is dataset.n_shards=1). Stopping 15 dataloader workers.`.

Result.
```console
wandb: Run summary:
wandb:      eval/time 1466.08443
wandb:       eval/wer 2.5685
wandb: eval/wer_ortho 98.80554
```
This value `eval/wer` is lower than the HuggingFace model card WER value 3.0%.

### Open AI Small model.

Small model card https://huggingface.co/openai/whisper-small#evaluation
WER 3.432%.

Evaluate Whisper model on A10 GPU.
* float32 precision.
* batch size set to 16 to mitigate out of memory.
* option `--language` is not set (if set `--language "en"` WER is 4.06815 %).
* option `-streaming` is not set (if set `--streaming` server disconnection seen after 35 iterations).
* environment variable TOKENIZERS_PARALLELISM=false to mitigate warning message.
```console
cd
cd distil-whisper/training

chmod +x scripts/run_short_form_eval_en_librispeech_small.sh

tmux  # Optional.

./scripts/run_short_form_eval_en_librispeech_small.sh
```

Result.
```console
wandb: Run summary:
wandb:      eval/time 297.52911
wandb:       eval/wer 3.44541
wandb: eval/wer_ortho 98.83787
```

### Open AI Small.en model.

Small model card https://huggingface.co/openai/whisper-small.en#evaluation
WER 3.0532%.

Evaluate Whisper model on A10 GPU.
* float32 precision.
* batch size set to 16 to mitigate out of memory.
* option `--language` is not set.
* option `-streaming` is not set (if set `--streaming` server disconnection seen after 35 iterations).
* environment variable TOKENIZERS_PARALLELISM=false to mitigate warning message.
```console
cd
cd distil-whisper/training

chmod +x scripts/run_short_form_eval_en_librispeech_small.en.sh

tmux  # Optional.

./scripts/run_short_form_eval_en_librispeech_small.en.sh
```

Result.
```console
wandb: Run summary:
wandb:      eval/time 285.10734
wandb:       eval/wer 3.05316
wandb: eval/wer_ortho 98.7675
```

### Open AI Tiny model.

Small model card https://huggingface.co/openai/whisper-tiny#evaluation
WER 7.547%.

Evaluate Whisper model on A10 GPU.
* float32 precision.
* batch size set to 16 to mitigate out of memory.
* option `--language` is not set (if set `--language "en"` WER is 8.23607%).
* option `-streaming` is not set (if set `--streaming` server disconnection seen after 35 iterations).
* environment variable TOKENIZERS_PARALLELISM=false to mitigate warning message.
```console
cd
cd distil-whisper/training

chmod +x scripts/run_short_form_eval_en_librispeech_tiny.sh

tmux  # Optional.

./scripts/run_short_form_eval_en_librispeech_tiny.sh
```

Result.
```console
wandb: Run summary:
wandb:      eval/time 49.7755
wandb:       eval/wer 7.56219
wandb: eval/wer_ortho 99.47885
```

### Open AI Tiny.en model.

Small model card https://huggingface.co/openai/whisper-tiny.en#evaluation
WER 5.656%.

Evaluate Whisper model on A10 GPU.
* float32 precision.
* batch size set to 16 to mitigate out of memory.
* option `--language` is not set.
* option `-streaming` is not set (if set `--streaming` server disconnection seen after 35 iterations).
* environment variable TOKENIZERS_PARALLELISM=false to mitigate warning message.
```console
cd
cd distil-whisper/training

chmod +x scripts/run_short_form_eval_en_librispeech_tiny.en.sh

tmux  # Optional.

./scripts/run_short_form_eval_en_librispeech_tiny.en.sh
```

Result.
```console
wandb: Run summary:
wandb:      eval/time 47.58256
wandb:       eval/wer 5.65561
wandb: eval/wer_ortho 99.12127
```

### Summary.

The table `eval/wer` values are generated by the
`run_short_form_eval_en_librispeech*` bash  scripts in this repo.

| model     | HF model card WER | eval/wer without option `--language` | eval/wer with `--language "en"`<sup>1</sup> |
| --------- |  ---------------- | ---------------- | -------------------- |
| OpenAI Large-v2 | [3.0004](https://huggingface.co/openai/whisper-large-v2#evaluation) | [2.5685](https://github.com/guynich/distil-whisper/blob/main/training/scripts/README.md#open-ai-large-v2-model) | [3.1683](https://github.com/guynich/distil-whisper/blob/main/training/scripts/README.md#open-ai-large-v2-model) |
| OpenAI Small | [3.4322](https://huggingface.co/openai/whisper-small#evaluation) | [3.44541](https://github.com/guynich/distil-whisper/blob/main/training/scripts/README.md#open-ai-small-model) | [4.0682](https://github.com/guynich/distil-whisper/blob/main/training/scripts/README.md#open-ai-small-model) |
| OpenAI Small.en | [3.0532](https://huggingface.co/openai/whisper-small.en#evaluation) | [3.0532](#open-ai-smallen-model) | - |
| OpenAI Tiny | [7.5471](https://huggingface.co/openai/whisper-tiny#evaluation) | [7.56219](https://github.com/guynich/distil-whisper/blob/main/training/scripts/README.md#open-ai-tiny-model) | [8.23607](https://github.com/guynich/distil-whisper/blob/main/training/scripts/README.md#open-ai-tiny-model) |
| OpenAI Tiny.en | [5.6556](https://huggingface.co/openai/whisper-tiny.en#evaluation) | [5.6556](#open-ai-tinyen-model) | - |

<sup>1</sup> The use of `language` option is not recommended with English speech recognition,
see https://github.com/guynich/distil-whisper/blob/da0159cf6ae1215ce29e772fb0d55c6f1409444f/training/run_short_form_eval.py#L222.  The results in this column have higher WER
values and are redundant.
