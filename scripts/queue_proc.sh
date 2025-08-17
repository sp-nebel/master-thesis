#!/bin/bash

# ==============================================================================
# File Pairing Script
#
# Description:
# This script searches for pairs of files in two separate directories based on a
# predefined key-value mapping of strings. For each pair found, it executes
# a user-defined command. This version assumes only one file will match each
# key and value. Jobs are queued in batches of 5 with 30-minute delays.
#
# Usage:
# ./your_script_name.sh /path/to/key/directory /path/to/value/directory
#
# ==============================================================================

# --- 1. Configuration: Define your string mapping ---
# Use the format [key]="value". The keys and values should be unique within the map.
# Add as many key-value pairs as you need.
declare -A ID_MAP=(
    ["_0_"]="_0_"
    ["_1_"]="_1_"
    ["_2_"]="_1_"
    ["_3_"]="_2_"
    ["_4_"]="_2_"
    ["_5_"]="_3_"
    ["_6_"]="_3_"
    ["_7_"]="_4_"
    ["_8_"]="_5_"
    ["_9_"]="_5_"
    ["_10_"]="_6_"
    ["_11_"]="_6_"
    ["_12_"]="_7_"
    ["_13_"]="_7_"
    ["_14_"]="_8_"
    ["_15_"]="_9_"
    ["_16_"]="_9_"
    ["_17_"]="_10_"
    ["_18_"]="_10_"
    ["_19_"]="_11_"
    ["_20_"]="_11_"
    ["_21_"]="_12_"
    ["_22_"]="_13_"
    ["_23_"]="_13_"
    ["_24_"]="_14_"
    ["_25_"]="_14_"
    ["_26_"]="_15_"
    ["_27_"]="_15_"
)

# --- 2. Argument and Directory Validation ---
# Check if the correct number of arguments (two directories) is provided.
if [ "$#" -ne 2 ]; then
    echo "Error: Invalid number of arguments."
    echo "Usage: $0 <key_directory> <value_directory>"
    exit 1
fi

KEY_DIR="$1"
VALUE_DIR="$2"

# Check if the provided paths are actually directories.
if [ ! -d "$KEY_DIR" ]; then
    echo "Error: Key directory '$KEY_DIR' not found or is not a directory."
    exit 1
fi

if [ ! -d "$VALUE_DIR" ]; then
    echo "Error: Value directory '$VALUE_DIR' not found or is not a directory."
    exit 1
fi

echo "Starting file search..."
echo "Key Directory:   $KEY_DIR"
echo "Value Directory: $VALUE_DIR"
echo "----------------------------------------"

# --- 3. Main Processing Loop ---
# Initialize job counter for batching
job_counter=0

# Iterate over each key defined in the ID_MAP associative array.
for key in "${!ID_MAP[@]}"; do
    value=${ID_MAP[$key]}

    echo "Processing mapping: Key=$key -> Value=$value"

    # Find the first file in the key directory containing the key in its name.
    # The `-quit` action tells `find` to exit after the first match, which is efficient.
    # To search recursively into subdirectories, remove the "-maxdepth 1" flag.
    key_file=$(find "$KEY_DIR" -maxdepth 1 -type f -name "*$key*" -print -quit)

    # Find the first file in the value directory containing the value in its name.
    value_file=$(find "$VALUE_DIR" -maxdepth 1 -type f -name "*$value*" -print -quit)

    # Check if we found a file for both the key and the value.
    if [[ -n "$key_file" && -n "$value_file" ]]; then
        echo "  => Match Found!"
        echo "     Key File:   $key_file"
        echo "     Value File: $value_file"

        # Calculate delay in minutes (30 minutes per batch of 5 jobs)
        batch_number=$((job_counter / 5))
        delay_minutes=$((batch_number * 30))
        
        # Format delay for sbatch --begin option
        if [ $delay_minutes -eq 0 ]; then
            begin_option=""
            echo "     Scheduling: Immediate"
        else
            begin_option="--begin=now+${delay_minutes}minutes"
            echo "     Scheduling: ${delay_minutes} minutes from now (batch $((batch_number + 1)))"
        fi

        # --- 4. YOUR CUSTOM COMMAND GOES HERE ---
        sbatch $begin_option ~/master-thesis/scripts/run_scripts/run_train_proc.sh "$value_file" "$key_file" "3B_layer${key}"
        
        # Increment job counter only for successful matches
        ((job_counter++))
        echo

    else
        # Provide feedback if one or both files were not found.
        if [[ -z "$key_file" ]]; then
            echo "  -> No file found for key '$key'."
        fi
        if [[ -z "$value_file" ]]; then
            echo "  -> No file found for value '$value'."
        fi
        echo # Add a newline for cleaner output
    fi
done

echo "----------------------------------------"
echo "Script finished."