import torch
import os
import argparse
import procrustes


def main(args):
    try:
        list1 = torch.load(args.file1_path)
        list2 = torch.load(args.file2_path)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Please ensure the paths are correct.")
        return
    except Exception as e:
        print(f"An error occurred loading tensors: {e}")
        return

    # Cast lists of tensors to float32
    list1 = [t.to(torch.float32) for t in list1]
    list2 = [t.to(torch.float32) for t in list2]

    
    try:
        list1 = torch.cat(list1, dim=0)
        list2 = torch.cat(list2, dim=0)
    except Exception as e:
        print(f"Error concatenating tensors from lists: {e}")
        print("Please ensure the loaded files contain lists of tensors with consistent hidden dimensions.")
        return

    print(f"Shape of concatenated list1: {list1.shape}")
    print(f"Shape of concatenated list2: {list2.shape}")

    indexes = torch.randperm(list1.shape[0])
    list1 = list1[indexes[:args.samples]]
    list2 = list2[indexes[:args.samples]]

    result = procrustes.orthogonal(list1.numpy(), list2.numpy())

    matrix = result.get('t')
    print(matrix.shape)

    print("Transformation Matrix:\n", matrix)

    output_dir = os.path.dirname(args.r_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    output_filename = os.path.join(output_dir, "procrustes_rotation_matrix.pt")
    torch.save(torch.from_numpy(matrix), output_filename)
    print(f"Saved rotation matrix to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Procrustes alignment on two tensors.")
    parser.add_argument("file1_path", type=str, help="Path to the first .pt file (list of tensors).")
    parser.add_argument("file2_path", type=str, help="Path to the second .pt file (list of tensors).")
    parser.add_argument("r_path", type=str, help="Path to save the matrix (.pt file).")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples (tokens) to use for alignment.")
    args = parser.parse_args()
    main(args)