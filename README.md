# Code for [Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](addlink)

## Requirements

Our implementation builds upon Huggingface Transformer version `v4.43.4`.

```shell
conda create -n aclsubmission python=3.9 
pip install -e .
pip install torch torchvision torchaudio
pip install deepspeed
pip install flash-attn --no-build-isolation
pip install datasets evalute peft
pip install skikit-learn hf_mtask_trainer 
```

The packages in our environment are listed in `environment.yml`.

## Change Overview

Our main modifications are:
 * `src/transformers/models/llama/modeling_llama.py` and `src/transformers/models/qwen2/modeling_qwen2.py` and `src/transformers/models/qwen2/modeling_qwen2.py` to enable the models for contrastive learning at middle layer
 * `src/transformers/trainer.py` to support alternate training between two losses\
 * `src/transformers/configuration_utils.py` to support reading in additional configuration parameters
 * `src/transformers/tokenization_utils_base.py` to support taking a pair of parallel sentences at once
 * `src/transformers/modeling_utils.py` to support tracking two different losses 

## Training 

Our training scripts adapted from [ParroT](https://github.com/wxjiao/ParroT).

The training and validation files (`--train_file $train_files` and `--validation_file $valid_files`) are expected in `jsonl` format, 
as shown in examples in `./data_example/train_baseline.json`
and `./data_example/train_baseline_with_alignment.json`.

The configuration files for DeepSpeed and LoRA are included in `./config`.

### Task-specific baseline training
```shell
bash ./scripts/train_baseline.sh
```

### Alternate training with alignment objective
```shell
bash ./scripts/train_with_alignment.sh
```

The changed/additional parameters are:
* `--alternate_training` activate alternate training
* `--contrastive_data_mode` set to 2 signal data format for alternate training
* `--additional_loss_layer $loss_layer` to specify which layer to add the loss (16 in our experiments)
* `--contrastive_loss_temperature $loss_temperature` defaults to 1.0
* `--distance_function $loss_distance_type` defaults to cosine 
* `--num_train_epochs` is set to doubled to 10 due to alternate training

