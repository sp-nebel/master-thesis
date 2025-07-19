import torch
import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple, Union

def _compare_single_tensor(tensor1: torch.Tensor, tensor2: torch.Tensor, tolerance: float, identifier: Union[str, int]):
    """
    Performs a detailed comparison of a single pair of tensors.
    This is a helper function for the main comparison logic.
    """
    print(f"\n>>> Comparing tensor: {identifier}")
    print(f"  Tensor 1 Info: shape={tensor1.shape}, dtype={tensor1.dtype}")
    print(f"  Tensor 2 Info: shape={tensor2.shape}, dtype={tensor2.dtype}")

    if tensor1.shape != tensor2.shape:
        print("  ❌ [RESULT] Tensors have different shapes. Cannot compare.")
        return

    if torch.equal(tensor1, tensor2):
        print("  ✅ [SUCCESS] Tensors are exactly identical.")
        return

    print("  ℹ️ [INFO] Tensors are NOT a perfect match. Analyzing deviation...")
    
    diff = torch.abs(tensor1 - tensor2)
    non_zero_mask = tensor1 != 0
    percentage_dev = torch.zeros_like(diff, dtype=torch.float32)
    percentage_dev[non_zero_mask] = (diff[non_zero_mask] / torch.abs(tensor1[non_zero_mask])) * 100
    num_different_elements = torch.count_nonzero(diff > tolerance)

    print("\n  --- Deviation Statistics ---")
    print(f"  Elements with difference > {tolerance}: {num_different_elements.item()} / {tensor1.numel()}")
    
    if num_different_elements > 0:
        print(f"  Mean Absolute Difference:   {torch.mean(diff).item():.6f}")
        print(f"  Maximum Absolute Difference:  {torch.max(diff).item():.6f}")
        
        if torch.any(non_zero_mask):
            differing_elements_mask = diff > tolerance
            mean_percent_dev = torch.mean(percentage_dev[differing_elements_mask]).item()
            max_percent_dev = torch.max(percentage_dev).item()
            print(f"  Mean Percentage Deviation (of differing elements): {mean_percent_dev:.4f}%")
            print(f"  Maximum Percentage Deviation: {max_percent_dev:.4f}%")
        else:
            print("  Percentage Deviation: N/A (Reference tensor is all zeros where differences occur)")
    print("-" * 25)

def _process_file_pair(file1_path: Path, file2_path: Path, tolerance: float):
    """
    Loads a single pair of .pt files and orchestrates their comparison.
    Handles files containing a single tensor, a list/tuple, or a dict.
    """
    print(f"\n{'='*60}\nComparing FILE:\n  1: {file1_path}\n  2: {file2_path}\n{'='*60}")
    try:
        data1 = torch.load(file1_path, map_location=torch.device('cpu'))
        data2 = torch.load(file2_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"❌ [ERROR] Could not load or read file pair. Error: {e}", file=sys.stderr)
        return

    # --- Universal Comparison Logic ---
    if isinstance(data1, dict) and isinstance(data2, dict):
        # --- Compare Dictionaries ---
        keys1, keys2 = set(data1.keys()), set(data2.keys())
        common_keys = sorted(list(keys1 & keys2))
        
        for key in common_keys:
            t1, t2 = data1[key], data2[key]
            if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
                _compare_single_tensor(t1, t2, tolerance, identifier=f"'{key}'")
            else:
                print(f"  ⚠️ [WARNING] Key '{key}' not a tensor in both files. Skipping.")
    else:
        # --- Compare Sequences or Single Tensors ---
        if not isinstance(data1, (list, tuple)): data1 = [data1]
        if not isinstance(data2, (list, tuple)): data2 = [data2]
        
        num_to_compare = min(len(data1), len(data2))
        if len(data1) != len(data2):
            print(f"⚠️ [WARNING] Files contain different number of tensors: {len(data1)} vs {len(data2)}.")
        
        for i in range(num_to_compare):
            t1, t2 = data1[i], data2[i]
            if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
                _compare_single_tensor(t1, t2, tolerance, identifier=f"at index {i}")
            else:
                print(f"  ⚠️ [WARNING] Item at index {i} is not a tensor in both files. Skipping.")

def compare_directories(dir1_path: str, dir2_path: str, tolerance: float):
    """
    Finds matching .pt files in two directories and compares them.
    """
    dir1 = Path(dir1_path)
    dir2 = Path(dir2_path)

    if not dir1.is_dir() or not dir2.is_dir():
        print("Error: One or both provided paths are not valid directories.", file=sys.stderr)
        sys.exit(1)

    # --- 1. Find all .pt files and their basenames ---
    files1 = {p.name: p for p in dir1.glob('*.pt')}
    files2 = {p.name: p for p in dir2.glob('*.pt')}

    # --- 2. Determine common and unique files ---
    filenames1 = set(files1.keys())
    filenames2 = set(files2.keys())

    common_files = sorted(list(filenames1 & filenames2))
    files_only_in_1 = sorted(list(filenames1 - filenames2))
    files_only_in_2 = sorted(list(filenames2 - filenames1))

    # --- 3. Print summary ---
    print("--- Directory Analysis ---")
    print(f"Found {len(common_files)} matching file(s) to compare.")
    if files_only_in_1:
        print(f"Files only in '{dir1_path}': {files_only_in_1}")
    if files_only_in_2:
        print(f"Files only in '{dir2_path}': {files_only_in_2}")
    
    if not common_files:
        print("\nNo matching files found to compare. Exiting.")
        sys.exit(0)

    # --- 4. Process each common file pair ---
    for filename in common_files:
        file1 = files1[filename]
        file2 = files2[filename]
        _process_file_pair(file1, file2, tolerance)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare all matching .pt files between two directories.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("dir1", help="Path to the first directory (the reference).")
    parser.add_argument("dir2", help="Path to the second directory.")
    parser.add_argument(
        "-t", "--tolerance",
        type=float,
        default=1e-8,
        help="Absolute tolerance. Differences below this value are ignored.\nDefault: 1e-8"
    )

    args = parser.parse_args()
    
    compare_directories(args.dir1, args.dir2, args.tolerance)
