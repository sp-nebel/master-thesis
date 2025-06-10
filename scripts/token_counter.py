import argparse
import json
import os
from transformers import AutoTokenizer
import tempfile # Added
import shutil   # Added



def main():
    parser = argparse.ArgumentParser(description="Counts tokens for a specified key in each line of a JSONL file and optionally creates a partial file when a token threshold is met.")
    parser.add_argument("jsonl_file", type=str, help="Path to the JSONL file.")
    parser.add_argument("model_name", type=str, help="Hugging Face model name for the tokenizer (e.g., 'bert-base-uncased').")
    parser.add_argument("key_to_tokenize", type=str, help="The key in the JSONL objects whose value should be tokenized.")
    parser.add_argument("--token_threshold", type=int, required=True, help="Token count threshold to trigger partial file creation.")
    parser.add_argument("--output_file", type=str, default=None, help="Optional: Path to the output file for the partial content. If not provided, it will be auto-generated based on the input file name and threshold.")
    
    args = parser.parse_args()

    if args.token_threshold <= 0:
        print("Warning: --token_threshold must be positive for partial file creation. Feature will be disabled.")
        # No partial file will be created if threshold is not positive.
        # The script will still count tokens.

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    total_tokens = 0
    total_lines = 0
    tokens_per_line = []
    # lines_buffer = [] # Removed: Replaced by temporary file
    output_file_created = False

    temp_file_handle = None
    temp_output_filename = None

    try:
        if args.token_threshold > 0:
            # Create a named temporary file; we'll manage its deletion/renaming.
            temp_file_handle = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False)
            temp_output_filename = temp_file_handle.name
        
        with open(args.jsonl_file, 'r', encoding='utf-8') as f:
            for line_content in f:
                if temp_file_handle: # If temp file is active (i.e., positive threshold)
                    temp_file_handle.write(line_content)
                
                total_lines += 1
                try:
                    data = json.loads(line_content)
                    text_to_tokenize = data.get(args.key_to_tokenize)
                    if text_to_tokenize is None:
                        print(f"Warning: Key '{args.key_to_tokenize}' not found in line: {line_content.strip()}")
                        continue
                    if not isinstance(text_to_tokenize, str):
                        print(f"Warning: Value for key '{args.key_to_tokenize}' is not a string in line: {line_content.strip()}")
                        continue
                    
                    tokens = tokenizer.encode(text_to_tokenize, add_special_tokens=True)
                    current_line_tokens = len(tokens)
                    tokens_per_line.append(current_line_tokens)
                    total_tokens += current_line_tokens

                    # Check for threshold and create file if met (and not already created)
                    # Only attempt if a positive threshold was set and not already created
                    if not output_file_created and args.token_threshold > 0 and total_tokens >= args.token_threshold:
                        output_filename = args.output_file
                        if not output_filename:
                            base, ext = os.path.splitext(args.jsonl_file)
                            output_filename = f"{base}_upto_{args.token_threshold}_tokens{ext}"
                        
                        try:
                            if temp_file_handle: # Should always be true if args.token_threshold > 0
                                temp_file_handle.close() # Close the temp file before renaming
                                temp_file_handle = None # Mark as closed
                                shutil.move(temp_output_filename, output_filename)
                                print(f"Created partial file: {output_filename}, containing {total_lines} lines (all lines processed up to threshold), with accumulated tokens ~{total_tokens}.")
                                output_file_created = True
                                temp_output_filename = None # Mark temp file as successfully handled (renamed)
                        except Exception as e_write_move: 
                            print(f"Error creating partial file from temporary storage to {output_filename}: {e_write_move}")
                            # temp_output_filename might still exist and will be caught by finally
                
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from line: {line_content.strip()}")
                except Exception as e:
                    print(f"An error occurred processing line: {line_content.strip()}. Error: {e}")

        # After the loop, if temp file was used but not renamed (e.g., threshold not met)
        if temp_file_handle: # i.e. it's still open
            temp_file_handle.close()
            temp_file_handle = None # Mark as closed for the finally block

    except FileNotFoundError:
        print(f"Error: File not found at {args.jsonl_file}")
        return
    except Exception as e:
        print(f"An unexpected error occurred reading the file: {e}")
        return
    finally:
        # Ensure temporary file is cleaned up if it still exists and wasn't handled
        if temp_file_handle: # If an error occurred before explicit close in try block
            temp_file_handle.close()
        if temp_output_filename and os.path.exists(temp_output_filename):
            try:
                os.remove(temp_output_filename)
                # print(f"Cleaned up unused temporary file: {temp_output_filename}") # Optional: can be verbose
            except OSError as e_remove:
                print(f"Warning: Could not remove temporary file {temp_output_filename}: {e_remove}")

    print(f"Processed {total_lines} lines.")
    # for i, num_tokens in enumerate(tokens_per_line):
    #     print(f"Line {i+1}: {num_tokens} tokens")
    print(f"Total tokens for key '{args.key_to_tokenize}' using tokenizer '{args.model_name}': {total_tokens}")

    if total_lines > 0 and not output_file_created and args.token_threshold > 0:
        print(f"Note: Token threshold of {args.token_threshold} was not reached. Total tokens accumulated: {total_tokens}.")
    elif output_file_created:
        pass # Message already printed

if __name__ == "__main__":
    main()