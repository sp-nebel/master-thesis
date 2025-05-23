import torch
import os
import argparse

def main(args):
    input_dir = args.input_dir
    layer_number = [] 
    token_numbers = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".pt"):
                continue
            temp_hs_3b = torch.load(os.path.join(root, file), map_location='cpu')
            total_layers_3B = temp_hs_3b.shape[1]
            num_tokens_3B = temp_hs_3b.shape[0]
            layer_number.append(total_layers_3B)
            token_numbers.append(num_tokens_3B)

    num_tokens_3B = sum(token_numbers)
    for token_number in token_numbers:
        print(f"Number of tokens in file: {token_number}")
    print(f"Number of tokens: {num_tokens_3B}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="run_outputs/hs_in_tokens_3B", help="Directory containing the input files.")
    args = parser.parse_args()
    main(args)