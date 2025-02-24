# Code for [Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](https://arxiv.org/pdf/2502.14830)

## Requirements

Our implementation builds upon Huggingface Transformer version `v4.43.4`.

```shell
conda create -n midalign python=3.9
conda activate midalign
pip install -e .
# pytorch
pip install torch torchvision torchaudio
# deepspeed
pip install deepspeed
# flash attention
pip install flash-attn --no-build-isolation
# other huggingface packags
pip install datasets evalute peft
# helper packages
pip install skikit-learn hf_mtask_trainer 
# for evaluation
pip install seqeval levenshtein
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

The training and validation files (`--train_file` and `--validation_file`) are expected in `jsonl` format, 
with the following example files:
* `./data_example/train_baseline.json` (for training baseline)
* `./data_example/train_baseline_with_alignment.json` (for training with alignment).

The configuration files for DeepSpeed and LoRA are included in `./config`.

### Task-specific baseline training
```shell
bash ./scripts/train_baseline.sh
```
<details>

```shell
#!/bin/bash
export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=#eno2np1 #eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29501}"

module load compiler/gnu/12
module load devel/cuda/12.0

HOST_NUM=1
INDEX=0

model_path="meta-llama/Meta-Llama-3-8B-Instruct"
train_files="./data_example/train_baseline.json" # replace by actual training data
valid_files="./data_example/train_baseline.json" # replace by actual validation data
train_bsz=32
eval_bsz=32
gradient_accumulation_steps=4
lora_config="./config/lora_config.json"
LR="5e-4"
OUTDIR="./test_run_outputs"
nproc_per_node=1 # number of GPUs used in training


torchrun --nnodes $HOST_NUM --node_rank $INDEX --nproc_per_node $nproc_per_node \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ./scripts/run_clm_lora.py \
    --deepspeed ./config/deepspeed_config.json \
    --bf16 True \
    --bf16_full_eval True \
    --model_name_or_path ${model_path} \
    --train_file $train_files \
    --validation_file $valid_files \
    --use_lora True \
    --lora_config $lora_config \
    --torch_dtype bfloat16 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 2 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size $train_bsz \
    --per_device_eval_batch_size $eval_bsz \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs 5 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "inverse_sqrt" \
    --logging_steps 10 \
    --block_size 2048 \
    --do_train \
    --eval_strategy "steps" \
    --eval_steps 200 \
    --eval_on_start \
    --streaming \
    --ddp_timeout 3600 \
    --seed 1 \
    --gradient_checkpointing True \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --patience 5 \
    --output_dir $OUTDIR \
    --disable_tqdm True --overwrite_output_dir 2>&1  | tee -a $OUTDIR/train.log
```
</details>

### Alternate training with alignment objective
```shell
bash ./scripts/train_with_alignment.sh
```

<details>

```shell
#!/bin/bash
export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=#eno2np1 #eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29501}"

module load compiler/gnu/12
module load devel/cuda/12.0

HOST_NUM=1
INDEX=0

model_path="meta-llama/Meta-Llama-3-8B-Instruct"
train_files="./data_example/train_baseline.json" # replace by actual training data
valid_files="./data_example/train_baseline.json" # replace by actual validation data
train_bsz=32
eval_bsz=32
gradient_accumulation_steps=4
lora_config="./config/lora_config.json"
LR="5e-4"
OUTDIR="./test_run_outputs"
nproc_per_node=1 # number of GPUs used in training
loss_layer=16
loss_temperature=0.1
loss_distance_type="cosine"


torchrun --nnodes $HOST_NUM --node_rank $INDEX --nproc_per_node $nproc_per_node \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ${train_path} \
    --deepspeed ./config/deepspeed_config.json \
    --bf16 True \
    --bf16_full_eval True \
    --model_name_or_path ${model_path} \
    --train_file $train_files \
    --validation_file $valid_files \
    --use_lora True \
    --lora_config ./config/lora_config.json \
    --torch_dtype bfloat16 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 1 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size $train_bsz \
    --per_device_eval_batch_size $eval_bsz \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs 10 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "inverse_sqrt" \
    --logging_steps 10 \
    --block_size 2048 \
    --do_train \
    --eval_strategy "steps" \
    --eval_steps 200 \
    --streaming \
    --ddp_timeout 3600 \
    --seed 1 \
    --gradient_checkpointing True \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --patience 5 \
    --output_dir $OUTDIR \
    --contrastive_data_mode 2 \
    --additional_loss_layer $loss_layer \
    --contrastive_loss_temperature $loss_temperature \
    --distance_function $loss_distance_type \
    --alternate_training \
    --disable_tqdm True --overwrite_output_dir 2>&1  | tee -a $OUTDIR/train.log
```

</details>

The changed/additional parameters are:
* `--alternate_training` activate alternate training
* `--contrastive_data_mode` set to 2 signal data format for alternate training
* `--additional_loss_layer $loss_layer` to specify which layer to add the loss (16 in our experiments)
* `--contrastive_loss_temperature $loss_temperature` defaults to 1.0
* `--distance_function $loss_distance_type` defaults to cosine 
* `--num_train_epochs` is set to doubled to 10 due to alternate training

## Data
The main experiment data in `.jsonl` format can be downloaded [here](https://bwsyncandshare.kit.edu/s/EDo3k3mibyejq6H).

## Inference and Evaluation
```shell
basemodel="meta-llama/Meta-Llama-3-8B-Instruct"
path2peftmodel="" # replace by path to finetuned model
lang="en" # replace by other langauge codes (see massive_lang_map in scripts/utils.py)

python -m scripts.run_inference_massive --model-name $basemodel \
                                        --peft-model-id $path2peftmodel \
                                        --lang $lang \ 
                                        --partition "test"
python -m scripts.eval_massive --pred-slots-file $path2output \
                               --lang $lang \
                               --partition "test"
```

(inference scripts for more datasets coming soon)


