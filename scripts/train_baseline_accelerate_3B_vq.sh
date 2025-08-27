#!/bin/bash


# --- Script Variables (Keep your existing variables) ---
model_path="meta-llama/Llama-3.2-3B-Instruct"
train_files="$TMPDIR/xnli_en_train.json" # replace by actual training data
valid_files="$TMPDIR/xnli_en_val.json" # replace by actual validation data
train_bsz=32 # Note: This is PER DEVICE batch size with accelerate/deepspeed
eval_bsz=16  # Note: This is PER DEVICE batch size
gradient_accumulation_steps=1
lora_config="$HOME/master-thesis/config/lora_config_v_q.json"
LR="5e-4"
OUTDIR="$TMPDIR/lora_model" # Changed output dir name example
mkdir -p $OUTDIR # Create output dir

accelerate launch \
    $HOME/master-thesis/scripts/run_clm_lora.py \
    --deepspeed $HOME/master-thesis/config/deepspeed_config.json \
    --bf16 True \
    --bf16_full_eval True \
    --model_name_or_path ${model_path} \
    --train_file $train_files \
    --validation_file $valid_files \
    --use_lora True \
    --lora_config $lora_config \
    --torch_dtype bfloat16 \
    --only_train_language_modeling True \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 2 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size $train_bsz \
    --per_device_eval_batch_size $eval_bsz \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs 1 \
    --torch_empty_cache_steps 200 \
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
    --disable_tqdm True \
    --overwrite_output_dir \
    2>&1 | tee -a $OUTDIR/train.log
