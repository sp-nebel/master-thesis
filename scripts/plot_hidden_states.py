import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

def get_rdm(activations_tensor):
    """Computes the RDM (1 - cosine_similarity) for a set of activations."""
    if activations_tensor.ndim == 1: # Single vector, make it 2D
        activations_tensor = activations_tensor.unsqueeze(0)
    if activations_tensor.shape[0] < 2: # Not enough items to compare
        return np.array([[0.0]]) # Or handle as an error/empty

    # Ensure tensor is on CPU and converted to numpy if it's not already
    if isinstance(activations_tensor, torch.Tensor):
        activations_np = activations_tensor.cpu().numpy()
    else:
        activations_np = activations_tensor

    sim_matrix = cosine_similarity(activations_np)
    rdm = 1 - sim_matrix
    return rdm

def compare_activations_rsa(hs_file_1B, hs_file_3B, token_indices, layer_idx_1B, layer_idx_3B):
    """
    Compares activations from two models for specific tokens and layers using RSA.
    token_indices: A list or slice of token indices to use for comparison.
    """
    hs_1B = torch.load(hs_file_1B, map_location='cpu')
    hs_3B = torch.load(hs_file_3B, map_location='cpu')

    # Extract activations for the specified tokens and layer
    # Shape: (num_selected_tokens, hidden_size)
    activations_1B_layer = hs_1B[token_indices, layer_idx_1B, :]
    activations_3B_layer = hs_3B[token_indices, layer_idx_3B, :]

    print(f"Shape of 1B activations for layer {layer_idx_1B}: {activations_1B_layer.shape}")
    print(f"Shape of 3B activations for layer {layer_idx_3B}: {activations_3B_layer.shape}")

    if activations_1B_layer.shape[0] < 2 or activations_3B_layer.shape[0] < 2:
        print("Not enough tokens selected to compute RDMs for comparison.")
        return None, None, None

    rdm_1B = get_rdm(activations_1B_layer)
    rdm_3B = get_rdm(activations_3B_layer)

    # Compare RDMs (flatten the upper triangle, excluding diagonal)
    rdm_1B_flat = rdm_1B[np.triu_indices_from(rdm_1B, k=1)]
    rdm_3B_flat = rdm_3B[np.triu_indices_from(rdm_3B, k=1)]

    if len(rdm_1B_flat) < 2 or len(rdm_3B_flat) < 2: # spearmanr needs at least 2 points
        print("Not enough elements in RDMs to compute correlation.")
        return rdm_1B, rdm_3B, None

    correlation, p_value = spearmanr(rdm_1B_flat, rdm_3B_flat)
    print(f"Spearman correlation between RDMs: {correlation:.4f} (p-value: {p_value:.4f})")

    return rdm_1B, rdm_3B, correlation

# --- Example Usage ---
# These paths and indices are placeholders
# Assume you have metadata to know total layers for each model
# And you've identified corresponding .pt files for the SAME input prompt
HS_FILE_1B_PROMPT1 = "run_outputs/hidden_states_1B/hidden_states_all_layers_0.pt" # Example
HS_FILE_3B_PROMPT1 = "run_outputs/hidden_states_3B/hidden_states_all_layers_0.pt" # Example

# --- Determine total layers (you'd get this from model config or hs shape) ---
temp_hs_1b = torch.load(HS_FILE_1B_PROMPT1, map_location='cpu')
total_layers_1B = temp_hs_1b.shape[1]
num_tokens_1B = temp_hs_1b.shape[0]
del temp_hs_1b

temp_hs_3b = torch.load(HS_FILE_3B_PROMPT1, map_location='cpu')
total_layers_3B = temp_hs_3b.shape[1]
num_tokens_3B = temp_hs_3b.shape[0]
del temp_hs_3b

# For demonstration, let's assume:
num_tokens_1B = 6 # From your script's max_new_tokens
num_tokens_3B = 6

# Compare first 5 generated tokens (if available)
common_token_indices = slice(0, min(5, num_tokens_1B, num_tokens_3B))


# Example: Compare an early layer (e.g., ~10% depth)
layer_1B_early = int(0.1 * total_layers_1B)
layer_3B_early = int(0.1 * total_layers_3B)
print(f"\nComparing early layers: 1B Layer {layer_1B_early} vs 3B Layer {layer_3B_early}")
rdm1_early, rdm3_early, corr_early = compare_activations_rsa(
    HS_FILE_1B_PROMPT1, HS_FILE_3B_PROMPT1,
    common_token_indices, layer_1B_early, layer_3B_early
)

# Example: Compare a middle layer (e.g., ~50% depth)
layer_1B_mid = int(0.5 * total_layers_1B)
layer_3B_mid = int(0.5 * total_layers_3B)
print(f"\nComparing middle layers: 1B Layer {layer_1B_mid} vs 3B Layer {layer_3B_mid}")
rdm1_mid, rdm3_mid, corr_mid = compare_activations_rsa(
    HS_FILE_1B_PROMPT1, HS_FILE_3B_PROMPT1,
    common_token_indices, layer_1B_mid, layer_3B_mid
)

# Example: Compare a late layer (e.g., ~90% depth)
layer_1B_late = int(0.9 * total_layers_1B)
layer_3B_late = int(0.9 * total_layers_3B)
print(f"\nComparing late layers: 1B Layer {layer_1B_late} vs 3B Layer {layer_3B_late}")
rdm1_late, rdm3_late, corr_late = compare_activations_rsa(
    HS_FILE_1B_PROMPT1, HS_FILE_3B_PROMPT1,
    common_token_indices, layer_1B_late, layer_3B_late
)

# --- Visualize an RDM pair (e.g., for middle layers) ---
if rdm1_mid is not None and rdm3_mid is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if rdm1_mid.shape[0] > 1 : # Check if RDM is more than a single point
        sns.heatmap(rdm1_mid, ax=axes[0], cmap="viridis", square=True, cbar=True)
        axes[0].set_title(f"1B Model - Layer {layer_1B_mid} RDM")
        axes[0].set_xlabel("Token Index")
        axes[0].set_ylabel("Token Index")

    if rdm3_mid.shape[0] > 1 :
        sns.heatmap(rdm3_mid, ax=axes[1], cmap="viridis", square=True, cbar=True)
        axes[1].set_title(f"3B Model - Layer {layer_3B_mid} RDM")
        axes[1].set_xlabel("Token Index")
        axes[1].set_ylabel("Token Index")

    plt.tight_layout()
    plt.savefig("rdm_comparison_middle_layers.png")
    print("\nSaved rdm_comparison_middle_layers.png")
    plt.show()