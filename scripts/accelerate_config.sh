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


accelerate config