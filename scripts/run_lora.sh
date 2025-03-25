#!/bin/bash
#SBATCH --partition=dev_gpu_4
#SBATCH --ntasks-per-node=30
#SBATCH --time=00:10:00
#SBATCH --mem=8gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=lora_job
#SBATCH --output=../lora_job.out


module load compiler/gnu/13.3
module load jupyter/ai/2024-11-29

# Check environment variables
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PATH: $PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source .env/bin/activate

# Run PyTorch check (before torchrun)
echo "PyTorch check BEFORE torchrun:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); if torch.cuda.is_available(): print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'CUDA device name: {torch.cuda.get_device_name(0)}'); print(f'CUDA version: {torch.version.cuda}'); print(f'cuDNN version: {torch.backends.cudnn.version()}')"


export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=#eno2np1 #eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29501}"

HOST_NUM=1
INDEX=0

model_path="meta-llama/Llama-3.2-3B-Instruct"
train_files="artifacts/xnli_en_train.json" # replace by actual training data
valid_files="artifacts/xnli_en_val.json" # replace by actual validation data
train_bsz=32
eval_bsz=32
gradient_accumulation_steps=1
lora_config="config/lora_config.json"
LR="5e-4"
OUTDIR="test_run_outputs"
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
    --num_train_epochs 1 \
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


deactivate
