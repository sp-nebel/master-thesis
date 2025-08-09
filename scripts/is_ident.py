import torch
import argparse
import sys

def load_matrices(matrix1_path, matrix2_path=None, device='cpu'):
    """
    Load one or two torch matrices from files.
    
    Args:
        matrix1_path (str): Path to first matrix file
        matrix2_path (str, optional): Path to second matrix file. Defaults to None.
        device (str): Device to load matrices on ('cpu' or 'cuda')
    
    Returns:
        If matrix2_path is None: (matrix1, None) or (None, None)
        If matrix2_path is not None: (matrix1, matrix2) or (None, None)
    """
    try:
        print(f"Loading matrix 1 from: {matrix1_path}")
        matrix1 = torch.load(matrix1_path, map_location=device)
        print(f"Matrix 1 shape: {matrix1.shape}")

        matrix2 = None
        if matrix2_path:
            print(f"Loading matrix 2 from: {matrix2_path}")
            matrix2 = torch.load(matrix2_path, map_location=device)
            print(f"Matrix 2 shape: {matrix2.shape}")
        
        return matrix1, matrix2
    except Exception as e:
        print(f"Error loading matrices: {e}")
        return None, None

def compute_errors_to_identity(matrix1, matrix2, device='cpu'):
    """
    Multiply two torch matrices and calculate errors to identity matrix.
    
    Args:
        matrix1 (torch.Tensor): First matrix
        matrix2 (torch.Tensor): Second matrix
        device (str): Device to perform computations on ('cpu' or 'cuda')
    
    Returns:
        dict: Dictionary with 'mse', 'mae', 'tae' between matrix product and identity matrix, or None if an error occurs.
    """
    try:
        # Check if matrices can be multiplied
        if matrix1.shape[1] != matrix2.shape[0]:
            raise ValueError(f"Cannot multiply matrices with shapes {matrix1.shape} and {matrix2.shape}")
        
        # Multiply matrices
        print("Computing matrix multiplication...")
        result = torch.matmul(matrix1, matrix2)
        print(f"Result shape: {result.shape}")
        
        # Check if result is square (required for identity comparison)
        if result.shape[0] != result.shape[1]:
            raise ValueError(f"Result matrix is not square ({result.shape}), cannot compare to identity")
        
        # Create identity matrix of same size
        identity = torch.eye(result.shape[0], device=device, dtype=result.dtype)
        
        # Calculate errors
        diff = result - identity
        mse = torch.mean(diff ** 2).item()
        mae = torch.mean(torch.abs(diff)).item()
        tae = torch.sum(torch.abs(diff)).item()
        
        return {'mse': mse, 'mae': mae, 'tae': tae}
        
    except Exception as e:
        print(f"Error during computation: {e}")
        return None

def compute_errors_to_permuted(matrix, device='cpu'):
    """
    Calculate errors between a matrix and a randomly permuted version of itself.
    
    Args:
        matrix (torch.Tensor): The input matrix.
        device (str): Device to perform computations on ('cpu' or 'cuda').
    
    Returns:
        dict: Dictionary with 'mse', 'mae', 'tae' between the matrix and its permuted version, or None if an error occurs.
    """
    try:
        # Flatten the matrix to a 1D tensor to permute all elements
        flat_matrix = matrix.flatten()
        
        # Create a random permutation of indices
        permuted_indices = torch.randperm(flat_matrix.nelement(), device=matrix.device)
        
        # Create the permuted flat tensor
        permuted_flat_matrix = flat_matrix[permuted_indices]
        
        # Reshape it back to the original matrix shape
        permuted_matrix = permuted_flat_matrix.view(matrix.shape)
        
        # Calculate errors
        diff = matrix - permuted_matrix
        mse = torch.mean(diff ** 2).item()
        mae = torch.mean(torch.abs(diff)).item()
        tae = torch.sum(torch.abs(diff)).item()
        
        return {'mse': mse, 'mae': mae, 'tae': tae}
        
    except Exception as e:
        print(f"Error during permuted MSE computation: {e}")
        return None

def compute_errors_between(matrix1, matrix2):
    """
    Calculate errors between two torch matrices.
    
    Args:
        matrix1 (torch.Tensor): First matrix
        matrix2 (torch.Tensor): Second matrix
    
    Returns:
        dict: Dictionary with 'mse', 'mae', 'tae' between the two matrices, or None if an error occurs.
    """
    try:
        # Check if matrices have the same shape
        if matrix1.shape != matrix2.shape:
            raise ValueError(f"Matrices must have the same shape, but got {matrix1.shape} and {matrix2.shape}")
        
        # Calculate errors
        diff = matrix1 - matrix2
        mse = torch.mean(diff ** 2).item()
        mae = torch.mean(torch.abs(diff)).item()
        tae = torch.sum(torch.abs(diff)).item()
        
        return {'mse': mse, 'mae': mae, 'tae': tae}
        
    except Exception as e:
        print(f"Error during MSE computation: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Compute MSE for torch matrices.')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], 
                       help='Device to use for computation (default: cpu)')
    
    parser.add_argument('--matrix1', help='Path to first matrix file (.pt)')
    parser.add_argument('--matrix2', help='Path to second matrix file (.pt)')

    parser.add_argument('--identity', action='store_true', help='Multiply matrix1 and matrix2 and compute MSE to identity. Requires --matrix1 and --matrix2.')
    parser.add_argument('--permuted', action='store_true', help='Compute MSE between matrix1 and its permuted version. Requires --matrix1.')
    parser.add_argument('--compare', action='store_true', help='Compute MSE between matrix1 and matrix2. Requires --matrix1 and --matrix2.')

    parser.add_argument('--mae', action='store_true', help='Calculate Mean Absolute Error.')
    parser.add_argument('--tae', action='store_true', help='Calculate Total Absolute Error.')

    args = parser.parse_args()
    
    # Check if CUDA is available if requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")

    matrix1, matrix2 = None, None

    if args.matrix1:
        matrix1, _ = load_matrices(args.matrix1, device=args.device)
        if matrix1 is None:
            sys.exit(1)
    
    if args.matrix2:
        matrix2, _ = load_matrices(args.matrix2, device=args.device)
        if matrix2 is None:
            sys.exit(1)

    if not (args.identity or args.permuted or args.compare):
        print("No computation mode selected. Use --identity, --permuted, or --compare.")
        parser.print_help()
        sys.exit(1)
    
    if args.identity:
        if matrix1 is None or matrix2 is None:
            print("Error: --identity requires --matrix1 and --matrix2.")
            sys.exit(1)
        
        errors = compute_errors_to_identity(matrix1, matrix2, args.device)
        
        if errors is not None:
            print(f"\n--- To Identity ---")
            print(f"MSE: {errors['mse']:.6f}")
            if args.mae:
                print(f"MAE: {errors['mae']:.6f}")
            if args.tae:
                print(f"TAE: {errors['tae']:.6f}")

    if args.permuted:
        if matrix1 is None:
            print("Error: --permuted requires --matrix1.")
            sys.exit(1)
        
        errors = compute_errors_to_permuted(matrix1, args.device)

        if errors is not None:
            print(f"\n--- To Permuted ---")
            print(f"MSE: {errors['mse']:.6f}")
            if args.mae:
                print(f"MAE: {errors['mae']:.6f}")
            if args.tae:
                print(f"TAE: {errors['tae']:.6f}")
    
    if args.compare:
        if matrix1 is None or matrix2 is None:
            print("Error: --compare requires --matrix1 and --matrix2.")
            sys.exit(1)
        
        errors = compute_errors_between(matrix1, matrix2)

        if errors is not None:
            print(f"\n--- Between Matrices ---")
            print(f"MSE: {errors['mse']:.6f}")
            if args.mae:
                print(f"MAE: {errors['mae']:.6f}")
            if args.tae:
                print(f"TAE: {errors['tae']:.6f}")

if __name__ == "__main__":
    main()