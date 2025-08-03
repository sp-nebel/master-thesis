#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=hook_3B
#SBATCH --output=logs/hook_3B.out
#SBATCH --time=00:30:00
#SBATCH --mem=128gb

# --- Job Configuration (Set from command-line argument) ---

# 1. Validate input argument
if [ -z "$1" ]; then
    echo "Error: No model tag provided. Please run as: sbatch $0 <model_tag>"
    echo "Example: sbatch $0 1B"
    exit 1
fi

MODEL_TAG=$1 # e.g., "1B" or "3B"

# 2. Set variables based on the tag
case "$MODEL_TAG" in
  "1B")
    MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
    JOB_TIME="00:20:00"
    JOB_MEM="32gb"
    ;;
  "3B")
    MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
    JOB_TIME="02:00:00"
    JOB_MEM="48gb"
    ;;
  *)
    echo "Error: Unknown model tag '$MODEL_TAG'. Supported tags: 1B, 3B"
    exit 1
    ;;
esac


# --- Script Execution ---

# Fail on first error
set -e

echo "Starting job for model tag: ${MODEL_TAG}"
echo "Model name: ${MODEL_NAME}"

# --- Environment Setup ---
module load compiler/gnu/14.2 devel/cuda/12.8 devel/python/3.12.3-gnu-14.2
source $HOME/master-thesis/.env/bin/activate

# --- Data and Output Prep ---
OUTPUT_DIR_TMP="$TMPDIR/${MODEL_TAG}"
OUTPUT_DIR_PERMANENT="$(ws_find ws_sascha)/hidden_states/${MODEL_TAG}_with_hook/input_layernorm/"
mkdir -p $OUTPUT_DIR_TMP
mkdir -p $OUTPUT_DIR_PERMANENT
rsync -avhP $HOME/master-thesis/artifacts/xnli_en_val_128k.json $TMPDIR/data.json

# --- Run Analysis ---
echo "Running analysis..."
python $HOME/master-thesis/scripts/collect_hs_with_hook.py \
        --base_model_name_or_path ${MODEL_NAME} \
        --peft_model_path $HOME/master-thesis/run_outputs/models/3B_tied_lora \
        --test_file $TMPDIR/data.json \
        --output_file ${OUTPUT_DIR_TMP} \
        --trust_remote_code \
        --torch_dtype bfloat16 \
        --batch_size 16 \
        --max_input_length 512 \
        --hook_module input_layernorm

# --- Copy Results Back ---
echo "Copying results to workspace..."
rsync -avhP ${OUTPUT_DIR_TMP}/ ${OUTPUT_DIR_PERMANENT}/

echo "Job finished successfully."
deactivate