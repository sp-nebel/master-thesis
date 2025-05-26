import torch
from sklearn.decomposition import PCA
import os
import glob

def load_hidden_states(file_path):
    """Loads hidden states from a .pt file."""
    # Assuming the .pt file directly contains the tensor
    # or a dictionary with a key like 'hidden_states'
    data = torch.load(file_path, map_location='cpu')
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, dict) and 'hidden_states' in data:
        return data['hidden_states']
    else:
        raise ValueError(f"Could not extract hidden states from {file_path}. "
                         "Please check the file structure.")

def reduce_dimensions_with_pca(hidden_states_list, n_components):
    """
    Performs PCA on a list of hidden states.

    Args:
        hidden_states_list (list of torch.Tensor): List of hidden states.
                                                 Each tensor is expected to be (seq_len, hidden_dim).
        n_components (int): Number of principal components to keep.

    Returns:
        list of torch.Tensor: List of dimension-reduced hidden states.
        pca (sklearn.decomposition.PCA): The fitted PCA object.
    """
    # Concatenate all hidden states along the sequence dimension for PCA fitting
    # Assuming each tensor in hidden_states_list is (seq_len, hidden_dim)
    # We want to fit PCA on all tokens, so we stack them: (total_tokens, hidden_dim)
    
    # Ensure all tensors in hidden_states_list are float32 before concatenation
    float32_hidden_states_list = [hs.float() for hs in hidden_states_list]
    all_hidden_states_stacked = torch.cat([hs.view(-1, hs.size(-1)) for hs in float32_hidden_states_list], dim=0)

    # Convert to numpy for scikit-learn
    # Ensure the stacked tensor is float32 before converting to numpy
    all_hidden_states_np = all_hidden_states_stacked.float().numpy()

    pca = PCA(n_components=n_components)
    pca.fit(all_hidden_states_np)

    reduced_hidden_states_list = []
    for hs in float32_hidden_states_list: # Use the float32 list here as well
        # Ensure hs is float32 before converting to numpy
        hs_np = hs.view(-1, hs.size(-1)).float().numpy()
        reduced_hs_np = pca.transform(hs_np)
        # Reshape back to (seq_len, n_components) if original was (seq_len, hidden_dim)
        # This assumes the first dimension of hs was seq_len
        reduced_hs_torch = torch.from_numpy(reduced_hs_np).float().view(hs.size(0), n_components)
        reduced_hidden_states_list.append(reduced_hs_torch)

    return reduced_hidden_states_list, pca

def main():
    # --- Configuration ---
    input_dir = "run_outputs/3B_base_hs/"  # Directory containing your .pt files
    output_dir = "run_outputs/pca_output/3B_base/" # Directory to save reduced hidden states
    n_pca_components = 3  # Desired number of dimensions after PCA
    file_pattern = "*all_layers_?.pt" # Pattern to match your .pt files
    # --- End Configuration ---

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_paths = glob.glob(os.path.join(input_dir, file_pattern))

    if not file_paths:
        print(f"No files found matching pattern {file_pattern} in directory {input_dir}")
        return

    all_original_hidden_states = []
    original_filenames = []

    print(f"Loading hidden states from {len(file_paths)} files...")
    for file_path in file_paths:
        try:
            hs_raw = load_hidden_states(file_path) # Load the tensor as is

            # Process hs_raw to ensure it's 2D (seq_len, hidden_dim) for PCA
            if hs_raw.ndim == 3:
                # Assuming shape is (seq_len, num_layers, hidden_dim)
                # e.g., torch.Size([145, 17, 2048]) means seq_len=145, num_layers=17, hidden_dim=2048
                # We'll select the hidden states from the last layer.
                # Change layer_index if you want a different layer (e.g., 0 for the first).
                # If your dimensions are (num_layers, seq_len, hidden_dim), use hs_raw[layer_index, :, :]
                layer_index = -1  # Selects the last layer
                hs = hs_raw[:, layer_index, :]
                print(f"  File {os.path.basename(file_path)}: original shape {hs_raw.shape}, "
                      f"selected layer {layer_index} (0-indexed from start, -1 for last). New shape for PCA: {hs.shape}")
            elif hs_raw.ndim == 2:
                # Already in the expected (seq_len, hidden_dim) format
                hs = hs_raw
                print(f"  File {os.path.basename(file_path)}: original shape {hs.shape}, using as is for PCA.")
            else:
                # Handle unexpected tensor dimensions
                raise ValueError(
                    f"Expected 2D (seq_len, hidden_dim) or 3D (seq_len, num_layers, hidden_dim) tensor, "
                    f"but got shape {hs_raw.shape} from {file_path}"
                )
            
            all_original_hidden_states.append(hs)
            original_filenames.append(os.path.basename(file_path))
            # The print statement about "Loaded {file_path} with shape {hs.shape}" from the original script
            # is now covered by the more detailed prints above.
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    if not all_original_hidden_states:
        print("No hidden states were successfully processed for PCA. Exiting.")
        return

    print(f"\nPerforming PCA to reduce to {n_pca_components} dimensions...")
    reduced_hidden_states_list, fitted_pca = reduce_dimensions_with_pca(
        all_original_hidden_states, n_pca_components
    )
    if fitted_pca.explained_variance_ratio_ is not None:
        print(f"Explained variance ratio by {n_pca_components} components: {fitted_pca.explained_variance_ratio_.sum():.4f}")
    else:
        print(f"Could not retrieve explained variance ratio. n_components might be too high or data problematic.")


    print("\nSaving reduced hidden states...")
    for i, reduced_hs in enumerate(reduced_hidden_states_list):
        original_filename = original_filenames[i]
        output_filename = f"pca_{n_pca_components}d_{original_filename}"
        output_path = os.path.join(output_dir, output_filename)
        torch.save(reduced_hs, output_path)
        print(f"Saved reduced hidden states for {original_filename} to {output_path} with shape {reduced_hs.shape}")

    # Optionally, save the PCA model itself
    # import joblib
    # pca_model_path = os.path.join(output_dir, "pca_model.joblib")
    # joblib.dump(fitted_pca, pca_model_path)
    # print(f"\nSaved PCA model to {pca_model_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()