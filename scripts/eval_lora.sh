#!/bin/bash
export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=#eno2np1 #eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29501}"

module load compiler/gnu/13.3
module load devel/cuda/12.0

HOST_NUM=1
INDEX=0

model_path="test_run_outputs/checkpoint-200"

valid_files="artifacts/xnli_en_val.json" # replace by actual validation data
eval_bsz=32
OUTDIR="./test_run_outputs"


torchrun --nnodes $HOST_NUM \
    ./scripts/run_clm_lora.py \
    --deepspeed ./config/deepspeed_config.json \
    --bf16 True \
    --bf16_full_eval True \
    --model_name_or_path ${model_path} \
    --validation_file $valid_files \
    --torch_dtype bfloat16 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 2 \
    --dataloader_pin_memory True \
    --per_device_eval_batch_size $eval_bsz \
    --block_size 2048 \
    --do_eval \
    --eval_strategy "steps" \
    --eval_steps 200 \
    --eval_on_start \
    --streaming \
    --ddp_timeout 3600 \
    --seed 1 \
    --metric_for_best_model "eval_loss" \
    --patience 5 \
    --output_dir $OUTDIR \
    --disable_tqdm True --overwrite_output_dir 2>&1  | tee -a $OUTDIR/train.log
