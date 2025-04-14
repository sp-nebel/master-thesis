#!/bin/bash

model_path="meta-llama/Llama-3.2-3B-Instruct" # replace by actual model path
adapter_path="test_run_outputs/checkpoint-5000" # replace by actual adapter path
train_files="artifacts/xnli_en_train.json" # replace by actual training data
valid_files="artifacts/xnli_en_val.json" # replace by actual validation data
eval_bsz=32
gradient_accumulation_steps=1
lora_config="./config/lora_config.json"
LR="5e-4"
OUTDIR="./test_run_outputs"
nproc_per_node=1 # number of GPUs used in training


python ./scripts/run_clm_lora.py \
    --bf16_full_eval True \
    --model_name_or_path ${model_path} \
    --load_lora_from $adapter_path \
    --metric_for_best_model "accuracy" \
    --validation_file $valid_files \
    --only_train_language_modeling True \
    --torch_dtype bfloat16 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 2 \
    --dataloader_pin_memory True \
    --per_device_eval_batch_size $eval_bsz \
    --torch_empty_cache_steps 200 \
    --block_size 2048 \
    --do_eval \
    --streaming \
    --ddp_timeout 3600 \
    --seed 1 \
    --output_dir $OUTDIR \
    --disable_tqdm True | tee -a $OUTDIR/eval.log
