import argparse
import json
from transformers import AutoTokenizer



def main():
    parser = argparse.ArgumentParser(description="Counts tokens for a specified key in each line of a JSONL file.")
    parser.add_argument("jsonl_file", type=str, help="Path to the JSONL file.")
    parser.add_argument("model_name", type=str, help="Hugging Face model name for the tokenizer (e.g., 'bert-base-uncased').")
    parser.add_argument("key_to_tokenize", type=str, help="The key in the JSONL objects whose value should be tokenized.")
    args = parser.parse_args()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    total_tokens = 0
    total_lines = 0
    tokens_per_line = []

    try:
        with open(args.jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                try:
                    data = json.loads(line)
                    text_to_tokenize = data.get(args.key_to_tokenize)
                    if text_to_tokenize is None:
                        print(f"Warning: Key '{args.key_to_tokenize}' not found in line: {line.strip()}")
                        continue
                    if not isinstance(text_to_tokenize, str):
                        print(f"Warning: Value for key '{args.key_to_tokenize}' is not a string in line: {line.strip()}")
                        continue
                    
                    tokens = tokenizer.encode(text_to_tokenize, add_special_tokens=True)
                    tokens_per_line.append(len(tokens))
                    total_tokens += len(tokens)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from line: {line.strip()}")
                except Exception as e:
                    print(f"An error occurred processing line: {line.strip()}. Error: {e}")


    except FileNotFoundError:
        print(f"Error: File not found at {args.jsonl_file}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    print(f"Processed {total_lines} lines.")
    for i, tokens in enumerate(tokens_per_line):
        print(f"Line {i+1}: {tokens} tokens")
    print(f"Total tokens for key '{args.key_to_tokenize}' using tokenizer '{args.model_name}': {total_tokens}")

if __name__ == "__main__":
    main()