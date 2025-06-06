import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from scipy.stats import pearsonr
import scipy.linalg
import os
import glob

def svcca(X, Y, threshold=0.99):
    """
    Compute SVCCA between two sets of representations
    X, Y: arrays of shape (n_samples, n_features)
    Returns: SVCCA similarity score and canonical correlations
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    # SVD for dimensionality reduction
    U_x, s_x, Vt_x = svd(X_centered.T, full_matrices=False)
    U_y, s_y, Vt_y = svd(Y_centered.T, full_matrices=False)
    
    # Keep components that explain threshold variance
    sum_sq_sx = np.sum(s_x**2)
    sum_sq_sy = np.sum(s_y**2)

    if sum_sq_sx < 1e-9: # Check for near-zero total variance for X
        print(f"    WARNING: X_centered has near-zero total variance ({sum_sq_sx}). Setting cutoff_x to 1.")
        cutoff_x = 1
    else:
        cumsum_x = np.cumsum(s_x**2) / sum_sq_sx
        cutoff_x = np.argmax(cumsum_x >= threshold) + 1
    
    if sum_sq_sy < 1e-9: # Check for near-zero total variance for Y
        print(f"    WARNING: Y_centered has near-zero total variance ({sum_sq_sy}). Setting cutoff_y to 1.")
        cutoff_y = 1
    else:
        cumsum_y = np.cumsum(s_y**2) / sum_sq_sy
        cutoff_y = np.argmax(cumsum_y >= threshold) + 1
    
    print(f"    Cutoff X: {cutoff_x} (out of {len(s_x)} singular values), Cutoff Y: {cutoff_y} (out of {len(s_y)} singular values)") # Log cutoffs
    
    # Reduce dimensions
    X_reduced = X_centered @ U_x[:, :cutoff_x]
    Y_reduced = Y_centered @ U_y[:, :cutoff_y]
    
    # Canonical Correlation Analysis
    # Center again after reduction
    X_reduced_centered = X_reduced - np.mean(X_reduced, axis=0)
    Y_reduced_centered = Y_reduced - np.mean(Y_reduced, axis=0)
    
    # Compute cross-correlation matrix
    Sigma_XX = X_reduced_centered.T @ X_reduced_centered / (X_reduced_centered.shape[0] - 1)
    Sigma_YY = Y_reduced_centered.T @ Y_reduced_centered / (Y_reduced_centered.shape[0] - 1)
    Sigma_XY = X_reduced_centered.T @ Y_reduced_centered / (X_reduced_centered.shape[0] - 1)
    
    # Solve generalized eigenvalue problem
    canonical_corrs = [0.0] # Default in case of issues
    try:
        # Regularization for numerical stability
        reg = 1e-6
        Sigma_XX_reg = Sigma_XX + reg * np.eye(Sigma_XX.shape[0])
        Sigma_YY_reg = Sigma_YY + reg * np.eye(Sigma_YY.shape[0])
        
        inv_sqrt_XX = np.linalg.inv(scipy.linalg.sqrtm(Sigma_XX_reg))
        inv_sqrt_YY = np.linalg.inv(scipy.linalg.sqrtm(Sigma_YY_reg))
        
        T = inv_sqrt_XX @ Sigma_XY @ inv_sqrt_YY
        _U_cca, s_cca, _Vt_cca = svd(T)
        
        # Canonical correlations are the singular values
        if len(s_cca) > 0:
            canonical_corrs = s_cca
        else:
            print("    WARNING: s_cca (canonical correlations) is empty.")
            canonical_corrs = [0.0] # Should not happen if cutoffs > 0 and CCA is valid
            
    except Exception as e_cca:
        print(f"    WARNING: CCA computation failed: {e_cca}. Using fallback (absolute Pearson correlation of first PCs).") # Log fallback
        if X_reduced.shape[1] > 0 and Y_reduced.shape[1] > 0:
            # pearsonr returns (correlation, p-value)
            corr_val, _ = pearsonr(X_reduced[:, 0].flatten(), Y_reduced[:, 0].flatten())
            canonical_corrs = [np.abs(corr_val)] # SVCCA is mean of non-negative canonical correlations
        else:
            print("    ERROR: Reduced dimensions are zero for fallback, cannot compute Pearson correlation.")
            canonical_corrs = [0.0]
    
    # Ensure correlations are within [0, 1] and calculate mean
    # Singular values (s_cca) should be non-negative. Pearson fallback is made non-negative with abs().
    # Clipping might be needed for numerical stability if values slightly exceed 1.0.
    final_similarity = np.mean(np.clip(canonical_corrs, 0, 1))
    return final_similarity, canonical_corrs

def load_all_hidden_states(file_path):
    """Load all hidden states from .pt file containing multiple layers"""
    print(f"Loading file: {file_path}")
    data = torch.load(file_path, map_location='cpu')
    
    print(f"Data type: {type(data)}")
    if torch.is_tensor(data):
        print(f"Tensor shape: {data.shape}")
        print(f"Tensor dtype: {data.dtype}")
        
        # Convert to float32 if BFloat16
        data = data.float()
        
        # Your data is [seq_len, num_layers, hidden_dim]
        if data.ndim == 3:
            seq_len, num_layers, hidden_dim = data.shape
            print(f"Detected seq_len={seq_len}, {num_layers} layers, hidden_dim={hidden_dim}")
            
            layers = {}
            for i in range(num_layers):
                # Each layer: [seq_len, hidden_dim] - extract all tokens for layer i
                layers[i] = data[:, i, :].numpy()  # Shape: [145, hidden_dim]
                
            print(f"Successfully loaded {len(layers)} layers")
            return layers
        else:
            print(f"Unexpected tensor shape: {data.shape}")
            return {0: data.numpy()}
            
    elif isinstance(data, dict):
        # Handle dictionary format
        layers = {}
        for key, value in data.items():
            print(f"Processing key: {key}")
            if torch.is_tensor(value):
                value = value.float()
                layers[key] = value.numpy()
        return layers
        
    else:
        raise ValueError(f"Unsupported data format in {file_path}")

def extract_layer_number_from_key(key):
    """Extract layer number from dictionary key"""
    import re
    key_str = str(key)
    match = re.search(r'(\d+)', key_str)
    return int(match.group(1)) if match else 0

def compute_svcca_between_models(model1_file, model2_file):
    """
    Compute SVCCA between all layers of two models
    """
    print(f"Loading {model1_file}")
    model1_layers = load_all_hidden_states(model1_file)
    
    print(f"Loading {model2_file}")
    model2_layers = load_all_hidden_states(model2_file)
    
    print(f"Model 1 has {len(model1_layers)} layers")
    print(f"Model 2 has {len(model2_layers)} layers")
    
    # Get common layers
    common_layers = set(model1_layers.keys()) & set(model2_layers.keys())
    print(f"Common layers: {sorted(common_layers)}")
    
    similarities = []
    layer_pairs = []
    
    # Compare all pairs of layers
    for layer1 in sorted(model1_layers.keys()):
        for layer2 in sorted(model2_layers.keys()):
            print(f"Computing SVCCA for layer {layer1} vs layer {layer2}")
            
            hidden1 = model1_layers[layer1]
            hidden2 = model2_layers[layer2]
            
            # Reshape if needed (handle different tensor shapes)
            if hidden1.ndim == 3:  # [batch, seq_len, hidden_dim]
                hidden1 = hidden1.reshape(-1, hidden1.shape[-1])
            elif hidden1.ndim == 2:  # [samples, hidden_dim]
                pass
            else:
                print(f"Warning: Unexpected shape for layer {layer1}: {hidden1.shape}")
                continue
            
            if hidden2.ndim == 3:  # [batch, seq_len, hidden_dim]
                hidden2 = hidden2.reshape(-1, hidden2.shape[-1])
            elif hidden2.ndim == 2:  # [samples, hidden_dim]
                pass
            else:
                print(f"Warning: Unexpected shape for layer {layer2}: {hidden2.shape}")
                continue
            
            # Ensure same number of samples
            min_samples = min(hidden1.shape[0], hidden2.shape[0])
            hidden1 = hidden1[:min_samples]
            hidden2 = hidden2[:min_samples]
            
            try:
                similarity, _ = svcca(hidden1, hidden2)
                similarities.append(similarity)
                layer_pairs.append((layer1, layer2))
                print(f"  SVCCA similarity: {similarity:.3f}")
            except Exception as e:
                print(f"  Error computing SVCCA: {e}")
                continue
    
    return similarities, layer_pairs

def plot_svcca_heatmap(similarities, layer_pairs, model1_name="Model 1", model2_name="Model 2"):
    """Create heatmap of SVCCA similarities"""
    if not similarities:
        print("No similarities to plot")
        return None
    
    # Create matrix
    all_layers1 = [pair[0] for pair in layer_pairs]
    all_layers2 = [pair[1] for pair in layer_pairs]
    
    max_layer1 = max(all_layers1) + 1 if all_layers1 else 1
    max_layer2 = max(all_layers2) + 1 if all_layers2 else 1
    
    similarity_matrix = np.zeros((max_layer1, max_layer2))
    
    for sim, (l1, l2) in zip(similarities, layer_pairs):
        similarity_matrix[l1, l2] = sim
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='viridis', 
                cbar_kws={'label': 'SVCCA Similarity'},
                xticklabels=[f"L{i}" for i in range(max_layer2)],
                yticklabels=[f"L{i}" for i in range(max_layer1)])
    
    plt.title(f'SVCCA Similarity: {model1_name} vs {model2_name}')
    plt.xlabel(f'{model2_name} Layers')
    plt.ylabel(f'{model1_name} Layers')
    plt.tight_layout()
    
    return plt.gcf()

def plot_diagonal_similarities(similarities, layer_pairs):
    """Plot similarities for corresponding layers (diagonal)"""
    diagonal_sims = []
    layers = []
    
    for sim, (l1, l2) in zip(similarities, layer_pairs):
        if l1 == l2:  # Corresponding layers
            diagonal_sims.append(sim)
            layers.append(l1)
    
    if not diagonal_sims:
        print("No diagonal similarities found")
        return None
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, diagonal_sims, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Layer')
    plt.ylabel('SVCCA Similarity')
    plt.title('SVCCA Similarity Across Corresponding Layers')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def main():
    # Define paths to your .pt files
    model1_file = "run_outputs/hidden_states/1B_tied_hs/hidden_states_input_tokens_all_layers_0.pt"  # Update this
    model2_file = "run_outputs/hidden_states/3B_tied_hs/hidden_states_input_tokens_all_layers_0.pt"  # Update this
    
    # Check if files exist
    if not os.path.exists(model1_file):
        print(f"Model 1 file not found: {model1_file}")
        # Try to find files in current directory
        pt_files = glob.glob("*.pt")
        if len(pt_files) >= 2:
            model1_file = pt_files[0]
            model2_file = pt_files[1]
            print(f"Using: {model1_file} and {model2_file}")
        else:
            print("Please update the file paths in the script")
            return
    
    # Compute SVCCA similarities
    similarities, layer_pairs = compute_svcca_between_models(model1_file, model2_file)
    
    if not similarities:
        print("No similarities computed")
        return
    
    # Create visualizations
    os.makedirs("plots", exist_ok=True)
    
    # Heatmap
    fig1 = plot_svcca_heatmap(similarities, layer_pairs)
    if fig1:
        fig1.savefig("plots/base_svcca_heatmap.png", dpi=300, bbox_inches='tight')
        print("Saved heatmap to plots/svcca_heatmap.png")
    
    # Diagonal plot
    fig2 = plot_diagonal_similarities(similarities, layer_pairs)
    if fig2:
        fig2.savefig("plots/base_svcca_diagonal.png", dpi=300, bbox_inches='tight')
        print("Saved diagonal plot to plots/svcca_diagonal.png")
    
    # Print summary statistics
    print(f"\nSVCCA Similarities:")
    print(f"Mean: {np.mean(similarities):.3f}")
    print(f"Std: {np.std(similarities):.3f}")
    print(f"Min: {np.min(similarities):.3f}")
    print(f"Max: {np.max(similarities):.3f}")
    

if __name__ == "__main__":
    main()