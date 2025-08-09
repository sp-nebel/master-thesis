import argparse
import json
import sys
import os


def read_jsonl(filepath, key):
    """Reads a JSON Lines file and extracts values for a specific key."""
    values = []
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    if key in data:
                        values.append(data[key])
                    else:
                        print(f"Warning: Key '{key}' not found in line {i+1} of {filepath}", file=sys.stderr)
                        values.append(None) # Append None or handle as needed
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON on line {i+1} in {filepath}", file=sys.stderr)
                    values.append(None) # Append None or handle as needed
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading {filepath}: {e}", file=sys.stderr)
        sys.exit(1)
    return values

def main(args):
    """Compares 'text' from file1 and 'prediction' from file2."""
    print(f"Comparing '{args.key1}' from {args.file1} with '{args.key2}' from {args.file2}")

    texts = read_jsonl(args.file1, args.key1)
    predictions = read_jsonl(args.file2, args.key2)

    if len(texts) != len(predictions):
        print(f"Error: Files have different number of lines ({len(texts)} vs {len(predictions)})", file=sys.stderr)
        sys.exit(1)

    if not texts:
        print("Files are empty or no data could be extracted.")
        sys.exit(0)

    matches = 0
    mismatches = 0

    print("\n--- Comparison Results ---")
    for i, (text, pred) in enumerate(zip(texts, predictions)):
        if text is None or pred is None:
            print(f"Line {i+1}: Skipping due to previous read error.")
            continue # Skip comparison if data wasn't read correctly

        # Optional: Normalize strings (e.g., lowercasing, stripping whitespace)
        text_norm = str(text).strip().lower()
        pred_norm = str(pred).strip().lower()

        if text_norm in pred_norm:
            matches += 1
        else:
            mismatches += 1
            if args.show_mismatches:
                print(f"Line {i+1}: MISMATCH")
                print(f"  File 1 ({args.key1}): '{text}'")
                print(f"  File 2 ({args.key2}): '{pred}'")
                print("-" * 20)

    print("\n--- Summary ---")
    total_lines = len(texts)
    print(f"Total lines compared: {total_lines}")
    print(f"Matches: {matches}")
    print(f"Mismatches: {mismatches}")
    if total_lines > 0:
        match_percentage = (matches / total_lines) * 100
        print(f"Match Percentage: {match_percentage:.2f}%")
    print("---------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare specific keys from two JSON Lines files.")
    parser.add_argument("file1", help="Path to the first JSON Lines file.")
    parser.add_argument("file2", help="Path to the second JSON Lines file.")
    parser.add_argument("--key1", default="text", help="Key to extract from the first file (default: 'text').")
    parser.add_argument("--key2", default="prediction", help="Key to extract from the second file (default: 'prediction').")
    parser.add_argument("--show-mismatches", action="store_true", help="Print details of mismatched lines.")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    main(args)