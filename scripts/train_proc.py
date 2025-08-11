import torch
import os
import argparse
import procrustes


def main(args):
    try:
        tensor1 = torch.load(args.file1_path).to(torch.float32)
        tensor2 = torch.load(args.file2_path).to(torch.float32)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Please ensure the paths are correct.")
        return
    except Exception as e:
        print(f"An error occurred loading tensors: {e}")
        return
    

    print(f"Shape of tensor1: {tensor1.shape}")
    print(f"Shape of tensor2: {tensor2.shape}")

    indexes = torch.randperm(tensor1.shape[0])

    if args.samples > 0 and args.samples < tensor1.shape[0]:
        tensor1 = tensor1[indexes[:args.samples]]
        tensor2 = tensor2[indexes[:args.samples]]

    result = procrustes.orthogonal(tensor1.numpy(), tensor2.numpy())

    matrix = result.get('t')
    print("Matrix shape: ", matrix.shape)

    print("Transformation Matrix:\n", matrix)

    output_dir = os.path.dirname(args.r_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    output_filename = os.path.join(output_dir, args.output_filename)
    torch.save(torch.from_numpy(matrix), output_filename)
    print(f"Saved transformation matrix to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Procrustes alignment on two tensors.")
    parser.add_argument("file1_path", type=str, help="Path to the first .pt file (list of tensors).")
    parser.add_argument("file2_path", type=str, help="Path to the second .pt file (list of tensors).")
    parser.add_argument("r_path", type=str, help="Path to save the matrix (.pt file).")
    parser.add_argument("--output_filename", type=str, help="Filename for the output matrix file.")
    parser.add_argument("--samples", type=int, default=-1, help="Number of samples (tokens) to use for alignment.")
    
    
    args = parser.parse_args()
    main(args)