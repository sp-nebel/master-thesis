#!/bin/bash

# A script to read a file line by line, perform a search-and-replace,
# and write the result to a new file.
# This method is efficient for very large files.

# --- Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -e

# --- Argument Validation ---
# Check if the correct number of arguments is provided.
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <input_file> <output_file> <string_to_find> <string_to_replace>"
    exit 1
fi

# Assign arguments to variables for clarity.
INPUT_FILE="$1"
OUTPUT_FILE="$2"
SEARCH_STRING="$3"
REPLACE_STRING="$4"

# Check if the input file exists and is readable.
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found."
    exit 1
fi

# Inform the user that the process is starting.
echo "Processing '$INPUT_FILE'..."
echo "Searching for '$SEARCH_STRING' and replacing with '$REPLACE_STRING'."
echo "Output will be saved to '$OUTPUT_FILE'."

# --- Main Processing Loop ---
# Create or clear the output file.
> "$OUTPUT_FILE"

# Read the input file line by line.
# 'while read -r line' prevents backslash interpretation and preserves leading/trailing whitespace.
# The input file is redirected to the loop using '< "$INPUT_FILE"'.
while IFS= read -r line; do
    # Use Bash's built-in parameter expansion for replacement.
    # ${line//search/replace} replaces all occurrences of 'search' with 'replace'.
    modified_line="${line//$SEARCH_STRING/$REPLACE_STRING}"

    # Append the modified line to the output file.
    echo "$modified_line" >> "$OUTPUT_FILE"
done < "$INPUT_FILE"

# --- Completion ---
echo "Processing complete. New file created at '$OUTPUT_FILE'."
