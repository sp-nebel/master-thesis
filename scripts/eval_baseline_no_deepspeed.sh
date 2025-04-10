#!/bin/bash
export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=#eno2np1 #eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

HOST_NUM=1
INDEX=0

model_path="meta-llama/Llama-3.2-3B-Instruct"
train_files="artifacts/xnli_en_train.json" # replace by actual training data
valid_files="artifacts/xnli_en_val.json" # replace by actual validation data
train_bsz=32
eval_bsz=32
gradient_accumulation_steps=1
lora_config="./config/lora_config.json"
LR="5e-4"
OUTDIR="./test_run_outputs"
nproc_per_node=1 # number of GPUs used in training


python ./scripts/run_clm_lora.py \
    --bf16_full_eval True \
    --load_lora_from $model_path \
    --validation_file $valid_files \
    --use_lora True \
    --lora_config $lora_config \
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
