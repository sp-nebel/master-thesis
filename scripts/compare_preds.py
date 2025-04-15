import sys

def labels_same(line1, line2):
    """Checks if two lines contain the same NLI label."""
    line1_lower = line1.lower()
    line2_lower = line2.lower()
    # Check for each label explicitly
    if "entailment" in line1_lower and "entailment" in line2_lower:
        return True
    elif "contradiction" in line1_lower and "contradiction" in line2_lower:
        return True
    elif "neutral" in line1_lower and "neutral" in line2_lower:
        return True
    else:
        return False

def read_two_files(filepath1, filepath2):
    """
    Reads two files line by line simultaneously, compares labels,
    and returns the total lines read and the count of matching labels.

    Args:
        filepath1 (str): Path to the first file.
        filepath2 (str): Path to the second file.

    Returns:
        tuple: (total_lines_read, correct_labels_count) or None if an error occurs.
    """
    try:
        # Use 'with' to ensure files are closed automatically, even if errors occur
        # Open both files at the same time
        with open(filepath1, 'r', encoding='utf-8') as file1, \
             open(filepath2, 'r', encoding='utf-8') as file2:
            # print(f"\n--- Comparing '{filepath1}' and '{filepath2}' ---") # Optional: Keep if you want this message

            line_num = 0
            correct_labels = 0
            # Use zip to iterate through both files line by line in parallel
            # zip stops when the shorter file is exhausted
            for line1, line2 in zip(file1, file2):
                line_num += 1
                if labels_same(line1, line2):
                    correct_labels += 1

            # print("\n--- Finished comparing common lines ---") # Optional: Keep if you want this message
            if line_num == 0:
                print(f"Warning: No common lines found or files '{filepath1}'/'{filepath2}' are empty.", file=sys.stderr)
                # Decide how to handle empty files: return 0,0 or raise error? Returning 0,0 for now.
                return 0, 0
            return line_num, correct_labels
            # Note: If files have different lengths, lines only present
            # in the longer file after the shorter one ends are not read by zip.

    except FileNotFoundError:
        print(f"Error: One or both files not found. Please check paths:", file=sys.stderr)
        print(f"  '{filepath1}'", file=sys.stderr)
        print(f"  '{filepath2}'", file=sys.stderr)
        sys.exit(1) # Exit with an error code
    except PermissionError:
        print(f"Error: Permission denied to read one or both files.", file=sys.stderr)
        sys.exit(1) # Exit with an error code
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1) # Exit with an error code

# --- Main part of the script ---
if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    # sys.argv[0] is the script name, sys.argv[1] is the first arg, etc.
    if len(sys.argv) != 3:
        # Print usage instructions to standard error
        print(f"Usage: python {sys.argv[0]} <filepath1> <filepath2>", file=sys.stderr)
        sys.exit(1) # Exit with an error code indicating improper usage

    # Get file paths from command-line arguments
    file_path_1 = sys.argv[1]
    file_path_2 = sys.argv[2]

    # Call the function to read and compare the files
    result = read_two_files(file_path_1, file_path_2)

    # The function exits on error, so if we reach here, result should be valid
    if result:
        line_num, correct_labels = result
        print(f"Compared files: '{file_path_1}' and '{file_path_2}'")
        print(f"Total common lines processed: {line_num}")
        print(f"Matching labels found: {correct_labels}")
        if line_num > 0:
             print(f"Agreement: {correct_labels / line_num:.2%}")
        else:
             print("Agreement: N/A (no lines processed)")
        print("\nScript finished successfully.")
