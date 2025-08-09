#!/bin/bash
#SBATCH --partition=gpu_a100_il
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=collect_hs
#SBATCH --output=logs/collect_hs_%j.out
#SBATCH --time=01:00:00
#SBATCH --mem=128gb

# --- Job Configuration (Set from command-line argument) ---

# 1. Validate input arguments
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Error: Missing arguments. Please run as: sbatch $0 <model_tag> <hook_module> <hook_type>"
    echo "Example: sbatch $0 1B o_proj post"
    exit 1
fi

MODEL_TAG=$1 # e.g., "1B" or "3B"
HOOK_MODULE=$2 # e.g., "o_proj"
HOOK_TYPE=$3 # e.g., "pre", "post", "both"

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

# --- Update job name and output file with arguments ---
#SBATCH --job-name=hook_${MODEL_TAG}_${HOOK_MODULE}_${HOOK_TYPE}
#SBATCH --output=logs/hook_${MODEL_TAG}_${HOOK_MODULE}_${HOOK_TYPE}.out

# --- Script Execution ---

# Fail on first error
set -e

echo "Starting job for model tag: ${MODEL_TAG}"
echo "Model name: ${MODEL_NAME}"
echo "Hook module: ${HOOK_MODULE}"
echo "Hook type: ${HOOK_TYPE}"

# --- Environment Setup ---
module load compiler/gnu/14.2 devel/cuda/12.8 devel/python/3.12.3-gnu-14.2
source $HOME/master-thesis/.env/bin/activate

# --- Data and Output Prep ---
OUTPUT_DIR_TMP="$TMPDIR/${MODEL_TAG}_${HOOK_MODULE}_${HOOK_TYPE}"
OUTPUT_DIR_PERMANENT="$(ws_find ws_sascha)/hidden_states/${MODEL_TAG}_with_hook/${HOOK_MODULE}/${HOOK_TYPE}/"
mkdir -p $OUTPUT_DIR_TMP
mkdir -p $OUTPUT_DIR_PERMANENT
rsync -avhP $HOME/master-thesis/artifacts/xnli_en_train_500k.json $TMPDIR/data.json

# --- Run Analysis ---
echo "Running analysis..."
python $HOME/master-thesis/scripts/collect_hs_with_hook.py \
        --base_model_name_or_path ${MODEL_NAME} \
        --test_file $TMPDIR/data.json \
        --output_file ${OUTPUT_DIR_TMP} \
        --trust_remote_code \
        --torch_dtype bfloat16 \
        --batch_size 16 \
        --max_input_length 512 \
        --hook_module ${HOOK_MODULE} \
        --hook_type ${HOOK_TYPE} \
        --save_batch_interval 40

# --- Copy Results Back ---
echo "Copying results to workspace..."
rsync -avhP ${OUTPUT_DIR_TMP}/* ${OUTPUT_DIR_PERMANENT}/

echo "Job finished successfully."
deactivate