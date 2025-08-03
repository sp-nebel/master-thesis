#!/bin/bash
#SBATCH --partition=dev_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=print_arch_1B # Will be updated by script
#SBATCH --output=logs/print_arch_1B.out # Will be updated by script
#SBATCH --time=00:05:00
#SBATCH --mem=32gb

# --- Job Configuration (Set from command-line arguments) ---

# 1. Validate input arguments
if [ -z "$1" ]; then
    echo "Error: No model tag provided."
    echo "Usage: sbatch $0 <model_tag> [path_to_peft_adapter]"
    echo "Example: sbatch $0 1B"
    echo "Example with PEFT: sbatch $0 3B /path/to/my/adapter"
    exit 1
fi

MODEL_TAG=$1
PEFT_MODEL_PATH=$2 # This can be empty if no PEFT model is provided

# 2. Set variables based on the model tag
case "$MODEL_TAG" in
  "1B")
    MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
    ;;
  "3B")
    MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
    ;;
  *)
    echo "Error: Unknown model tag '$MODEL_TAG'. Supported tags: 1B, 3B"
    exit 1
    ;;
esac

# Update job name and log file based on model tag
#SBATCH --job-name=print_arch_${MODEL_TAG}
#SBATCH --output=logs/print_arch_${MODEL_TAG}.out


# --- Script Execution ---

# Fail on first error
set -e

echo "Starting job for model tag: ${MODEL_TAG}"
echo "Base model name: ${MODEL_NAME}"

# --- Environment Setup ---
module load compiler/gnu/14.2 devel/cuda/12.8 devel/python/3.12.3-gnu-14.2
source $HOME/master-thesis/.env/bin/activate

# --- Output Prep ---
OUTPUT_DIR="$(ws_find ws_sascha)/model_architectures/"
mkdir -p $OUTPUT_DIR

# Construct a descriptive output filename
if [ -n "$PEFT_MODEL_PATH" ]; then
    ADAPTER_NAME=$(basename "$PEFT_MODEL_PATH")
    OUTPUT_FILENAME="${MODEL_TAG}_with_${ADAPTER_NAME}.txt"
else
    OUTPUT_FILENAME="${MODEL_TAG}_base_model.txt"
fi
OUTPUT_FILE_PATH="${OUTPUT_DIR}/${OUTPUT_FILENAME}"

echo "Output will be saved to: ${OUTPUT_FILE_PATH}"

# --- Build Python Command ---
CMD="python $HOME/master-thesis/scripts/print_arch.py \
        --base_model_name_or_path ${MODEL_NAME} \
        --output_file ${OUTPUT_FILE_PATH} \
        --trust_remote_code \
        --torch_dtype bfloat16"

if [ -n "$PEFT_MODEL_PATH" ]; then
    echo "PEFT adapter path: ${PEFT_MODEL_PATH}"
    CMD="$CMD --peft_model_path ${PEFT_MODEL_PATH}"
    # Optionally add merging if you want to see that architecture too
fi

# --- Run Analysis ---
echo "Running analysis..."
echo "Executing command: $CMD"
eval $CMD

rsync -avhP ${OUTPUT_FILE_PATH} $HOME/master-thesis/run_outputs/model_architectures/

echo "Job finished successfully."
deactivate
