import argparse
from matplotlib import pyplot as plt
import torch
import os

def find_optimal_rank(S, energy_threshold=0.99):
    """
    Finds the optimal rank k that captures a certain percentage of the total energy.
    Energy is defined as the sum of squared singular values.

    Args:
        S (torch.Tensor): A 1D tensor of singular values, sorted in descending order.
        energy_threshold (float): The desired percentage of energy to capture (e.g., 0.99 for 99%).

    Returns:
        int: The smallest rank k that captures at least the specified energy threshold.
    """
    squared_singular_values = S.pow(2)
    total_energy = torch.sum(squared_singular_values)
    cumulative_energy = torch.cumsum(squared_singular_values, dim=0)
    energy_fraction = cumulative_energy / total_energy
    
    # Find the first rank where the captured energy exceeds the threshold
    # torch.where returns a tuple of tensors; we need the first element.
    # Add 1 because tensor indices are 0-based.
    optimal_k = torch.where(energy_fraction >= energy_threshold)[0][0].item() + 1
    
    return optimal_k

def print_statistics(tensor, name):
    """Calculates and prints summary statistics for a tensor."""
    mean_val = tensor.mean().item()
    std_val = tensor.std().item()
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    
    print(f"--- Statistics for {name} ---")
    print(f"Mean:         {mean_val:.4f}")
    print(f"Std Dev:      {std_val:.4f}")
    print(f"Min:          {min_val:.4f}")
    print(f"Max:          {max_val:.4f}\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(0)

    # Load the torch matrices
    small = torch.load(args.small_hidden_states, map_location=device).to(torch.float32)
    big = torch.load(args.big_hidden_states, map_location=device).to(torch.float32)

    print(f"Small matrix shape: {small.shape}")
    print(f"Big matrix shape: {big.shape}")
    print_statistics(small, "Small Hidden States")
    print_statistics(big, "Big Hidden States")

    # Apply sampling if specified
    if args.samples > 0 and args.samples < small.shape[0]:
        indexes = torch.randperm(small.shape[0])[:args.samples]
        small = small[indexes]
        big = big[indexes]
        print(f"Sampled to {args.samples} rows")

    # --- CHANGE: Center the data (crucial for covariance) ---
    small_centered = small - small.mean(dim=0, keepdim=True)
    big_centered = big - big.mean(dim=0, keepdim=True)
    print("Centered the hidden state matrices.")

    # --- CHANGE: Compute the cross-covariance matrix ---
    # The shape will be (small_dim, big_dim)
    cross_covariance_matrix = small_centered.T @ big_centered
    print(f"Cross-covariance matrix shape: {cross_covariance_matrix.shape}")

    # --- CHANGE: Perform a single SVD on the cross-covariance matrix ---
    # U corresponds to the small model's space, Vh.T corresponds to the big model's space.
    U, S, Vh = torch.linalg.svd(cross_covariance_matrix, full_matrices=False)
    
    # --- CHANGE: The plotting logic now uses the singular values from the cross-covariance SVD ---
    if args.plot_singular_values:
        plt.figure(figsize=(10, 6))
        plt.plot(S.cpu().numpy())
        plt.yscale('log')
        plt.title(f'Singular Value Spectrum of Cross-Covariance (Layer {args.layer})')
        plt.xlabel('Singular Value Index')
        plt.ylabel('Singular Value (log scale)')
        plt.grid(True)
        os.makedirs(args.output_dir, exist_ok=True)
        plt.savefig(os.path.join(args.output_dir, f'singular_values_layer_{args.layer}.png'))
        plt.close()
        print("Singular value plot saved. Exiting.")
        return
    
    # --- CHANGE: Dynamically determine rank if args.rank is 0 ---
    if args.rank == 0:
        rank = find_optimal_rank(S, energy_threshold=0.99)
        print(f"Dynamically determined rank to be {rank} to capture 99% of the energy.")
    else:
        rank = args.rank
        ideal_rank = find_optimal_rank(S, energy_threshold=0.99)
        print(f"Ideal rank to capture 99% energy: {ideal_rank}")
        print(f"Using fixed rank: {rank}")


    # Truncate the singular vectors U and V to the desired rank
    U_truncated = U[:, :rank]
    # Vh is V.T, so we get V by transposing Vh
    V_truncated = Vh.T[:, :rank]
    print(f"Truncated U to shape: {U_truncated.shape}")
    print(f"Truncated V to shape: {V_truncated.shape}")


    # --- CHANGE: Use the correct formulas to compute the mapping matrices ---
    
    # P_up maps from the small space to the big space.
    # Formula: V_k @ U_k.T
    # A vector h_s (small_dim,) is mapped via: h_l = P_up @ h_s
    P_up = V_truncated @ U_truncated.T
    
    # P_down maps from the big space to the small space.
    # Formula: U_k @ V_k.T
    # A vector h_l (big_dim,) is mapped via: h_s = P_down @ h_l
    P_down = U_truncated @ V_truncated.T

    # The shapes are transposed compared to the original script because the mapping
    # is applied via matrix-vector multiplication (e.g., P_up @ h_s).
    print(f"Shape of P_up (small -> big): {P_up.shape}") # Should be (big_dim, small_dim)
    print(f"Shape of P_down (big -> small): {P_down.shape}") # Should be (small_dim, big_dim)

    os.makedirs(args.output_dir, exist_ok=True)

    results = {
        'P_up': P_up.cpu(),
        'P_down': P_down.cpu(),
        'U_truncated': U_truncated.cpu(), # Corresponds to the smaller model's basis
        'V_truncated': V_truncated.cpu()  # Corresponds to the larger model's basis
    }

    output_file = os.path.join(args.output_dir, f"svd_layer_{args.layer}.pt")
    torch.save(results, output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SVD-based mappings between hidden states.")
    parser.add_argument("small_hidden_states", help="Path to .pt file for small model's hidden states")
    parser.add_argument("big_hidden_states", help="Path to .pt file for big model's hidden states")
    parser.add_argument("--rank", "-r", type=int, default=1300, help="Rank for SVD truncation. Set to 0 to dynamically find rank for 99%% energy. (default: 1300)")
    parser.add_argument("--samples", "-s", type=int, default=10000, help="Number of samples to use (default: 10000, 0 for all)")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory to save mapping files")
    parser.add_argument("--layer", "-l", type=int, required=True, help="Layer number (for file naming)")
    parser.add_argument("--plot_singular_values", action="store_true", help="Only plot singular values and exit")
    args = parser.parse_args()
    main()