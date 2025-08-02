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

def compute_mse_to_identity(matrix1, matrix2, device='cpu'):
    """
    Multiply two torch matrices and calculate MSE to identity matrix.
    
    Args:
        matrix1 (torch.Tensor): First matrix
        matrix2 (torch.Tensor): Second matrix
        device (str): Device to perform computations on ('cpu' or 'cuda')
    
    Returns:
        float: MSE between matrix product and identity matrix, or None if an error occurs.
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
        
        # Calculate MSE
        mse = torch.mean((result - identity) ** 2)
        
        return mse.item()
        
    except Exception as e:
        print(f"Error during computation: {e}")
        return None

def compute_mse_to_permuted(matrix, device='cpu'):
    """
    Calculate MSE between a matrix and a randomly permuted version of itself.
    
    Args:
        matrix (torch.Tensor): The input matrix.
        device (str): Device to perform computations on ('cpu' or 'cuda').
    
    Returns:
        float: MSE between the matrix and its permuted version, or None if an error occurs.
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
        
        # Calculate MSE
        mse = torch.mean((matrix - permuted_matrix) ** 2)
        
        return mse.item()
        
    except Exception as e:
        print(f"Error during permuted MSE computation: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Compute MSE for torch matrices.')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], 
                       help='Device to use for computation (default: cpu)')
    
    subparsers = parser.add_subparsers(dest='mode', required=True, help='computation mode')

    # Subparser for identity MSE
    parser_identity = subparsers.add_parser('identity', help='Multiply two matrices and compute MSE to identity')
    parser_identity.add_argument('matrix1', help='Path to first matrix file (.pt)')
    parser_identity.add_argument('matrix2', help='Path to second matrix file (.pt)')

    # Subparser for permuted MSE
    parser_permuted = subparsers.add_parser('permuted', help='Compute MSE between a matrix and its permuted version')
    parser_permuted.add_argument('matrix', help='Path to matrix file (.pt)')

    args = parser.parse_args()
    
    # Check if CUDA is available if requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    if args.mode == 'identity':
        matrix1, matrix2 = load_matrices(args.matrix1, args.matrix2, args.device)
        
        if matrix1 is None or matrix2 is None:
            sys.exit(1)

        mse = compute_mse_to_identity(matrix1, matrix2, args.device)
        
        if mse is not None:
            print(f"\nMSE to identity matrix: {mse:.6f}")
        else:
            sys.exit(1)
    
    elif args.mode == 'permuted':
        matrix, _ = load_matrices(args.matrix, device=args.device)

        if matrix is None:
            sys.exit(1)
        
        mse = compute_mse_to_permuted(matrix, args.device)

        if mse is not None:
            print(f"\nMSE to permuted matrix: {mse:.6f}")
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()