import torch
from sklearn.decomposition import PCA
import os
import glob
import argparse # Added for command-line arguments
# import joblib # Uncomment to save PCA models

def load_hidden_states(file_path):
    """Loads hidden states from a .pt file."""
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
    Performs PCA on a list of hidden states for a specific layer.

    Args:
        hidden_states_list (list of torch.Tensor): List of hidden states.
                                                 Each tensor is (seq_len, hidden_dim).
        n_components (int): Number of principal components to keep.

    Returns:
        list of torch.Tensor: List of dimension-reduced hidden states.
        pca (sklearn.decomposition.PCA): The fitted PCA object.
    """
    if not hidden_states_list:
        return [], PCA(n_components=n_components) # Return empty and unfitted PCA

    # Ensure all tensors are float32 before concatenation
    float32_hidden_states_list = [hs.float() for hs in hidden_states_list]
    
    # Concatenate all hidden states for PCA fitting: (total_tokens_for_this_layer, hidden_dim)
    all_hidden_states_stacked = torch.cat([hs.view(-1, hs.size(-1)) for hs in float32_hidden_states_list], dim=0)

    all_hidden_states_np = all_hidden_states_stacked.numpy()

    pca = PCA(n_components=n_components)
    pca.fit(all_hidden_states_np)

    reduced_hidden_states_list = []
    for hs in float32_hidden_states_list: # Use the float32 list
        hs_np = hs.view(-1, hs.size(-1)).numpy() # hs is already float32
        reduced_hs_np = pca.transform(hs_np)
        # Reshape back to (seq_len, n_components)
        reduced_hs_torch = torch.from_numpy(reduced_hs_np).float().view(hs.size(0), n_components)
        reduced_hidden_states_list.append(reduced_hs_torch)

    return reduced_hidden_states_list, pca

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Perform PCA on hidden states from .pt files.")
    parser.add_argument("--input_dir", type=str, default="run_outputs/hidden_states/3B_tied_hs",
                        help="Directory containing .pt files with hidden states.")
    parser.add_argument("--output_dir", type=str, default="run_outputs/pca_output/3B_tied_all_layers/",
                        help="Directory to save reduced hidden states.")
    parser.add_argument("--n_pca_components", type=int, default=3,
                        help="Desired number of dimensions after PCA.")
    parser.add_argument("--file_pattern", type=str, default="*all_layers_?.pt",
                        help="Pattern to match .pt files (e.g., '*all_layers_?.pt').")
    
    args = parser.parse_args()

    # Use parsed arguments
    input_dir = args.input_dir
    output_dir = args.output_dir
    n_pca_components = args.n_pca_components
    file_pattern = args.file_pattern
    # --- End Argument Parsing ---


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    file_paths = glob.glob(os.path.join(input_dir, file_pattern))

    if not file_paths:
        print(f"No files found matching pattern '{file_pattern}' in directory '{input_dir}'")
        return

    # Structure to hold hidden states: {layer_idx: [(filename, tensor), ...]}
    all_layer_specific_data = {}

    print(f"Loading hidden states from {len(file_paths)} files and organizing by layer...")
    num_layers_global = None

    for file_path in file_paths:
        try:
            hs_raw = load_hidden_states(file_path) # Load the tensor

            if hs_raw.ndim == 3:
                # Assuming shape is (seq_len, num_layers, hidden_dim)
                # e.g., torch.Size([145, 17, 2048]) means seq_len=145, num_layers=17, hidden_dim=2048
                seq_len, num_layers_in_file, hidden_dim = hs_raw.shape
                
                if num_layers_global is None:
                    num_layers_global = num_layers_in_file
                elif num_layers_global != num_layers_in_file:
                    print(f"  Warning: File {os.path.basename(file_path)} has {num_layers_in_file} layers, "
                          f"expected {num_layers_global} based on first file. "
                          f"Processing up to min(num_layers_global, num_layers_in_file).")
                    # This script will iterate up to num_layers_in_file for this file,
                    # and PCA will be fitted per layer_idx based on available data.

                print(f"  File {os.path.basename(file_path)}: original shape {hs_raw.shape}")

                for l_idx in range(num_layers_in_file):
                    layer_tensor = hs_raw[:, l_idx, :].float() # Shape: (seq_len, hidden_dim)
                    if l_idx not in all_layer_specific_data:
                        all_layer_specific_data[l_idx] = []
                    all_layer_specific_data[l_idx].append(
                        (os.path.basename(file_path), layer_tensor)
                    )
            elif hs_raw.ndim == 2:
                print(f"  File {os.path.basename(file_path)}: shape {hs_raw.shape}. "
                      "Skipping 2D file as layer index is ambiguous for layer-wise PCA processing of 3D tensors.")
                continue # Skip this file for layer-wise PCA
            else:
                raise ValueError(
                    f"Expected 3D (seq_len, num_layers, hidden_dim) tensor, "
                    f"but got shape {hs_raw.shape} from {file_path}"
                )
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    if not all_layer_specific_data:
        print("No 3D hidden states were successfully processed and organized by layer. Exiting.")
        return

    print(f"\nFound data for layers: {sorted(all_layer_specific_data.keys())}")
    print(f"Performing PCA to reduce to {n_pca_components} dimensions for each layer...")

    for layer_idx in sorted(all_layer_specific_data.keys()):
        print(f"\n--- Processing Layer {layer_idx} ---")
        
        layer_data_tuples = all_layer_specific_data[layer_idx] # List of (filename, tensor)
        
        if not layer_data_tuples:
            print(f"No data found for layer {layer_idx} after initial loading. Skipping.")
            continue

        current_layer_hidden_states = [item[1] for item in layer_data_tuples]
        current_layer_filenames = [item[0] for item in layer_data_tuples]

        if not current_layer_hidden_states:
            print(f"No hidden state tensors to process for layer {layer_idx}. Skipping.")
            continue
            
        print(f"Found {len(current_layer_hidden_states)} file(s) with data for layer {layer_idx}.")
        
        reduced_states_for_layer_list, fitted_pca_for_layer = reduce_dimensions_with_pca(
            current_layer_hidden_states, n_pca_components
        )
        
        if fitted_pca_for_layer.n_components_ is None or not hasattr(fitted_pca_for_layer, 'explained_variance_ratio_'):
             print(f"Layer {layer_idx} - PCA not fitted or explained variance ratio not available (possibly no data or n_components issue).")
        elif fitted_pca_for_layer.explained_variance_ratio_ is not None:
            print(f"Layer {layer_idx} - Explained variance by {fitted_pca_for_layer.n_components_} components: "
                  f"{fitted_pca_for_layer.explained_variance_ratio_.sum():.4f}")
        else: # Should be covered by the above, but as a fallback
            print(f"Layer {layer_idx} - Could not retrieve explained variance ratio.")


        print(f"Saving reduced hidden states for Layer {layer_idx}...")
        for i, reduced_hs_tensor in enumerate(reduced_states_for_layer_list):
            original_filename = current_layer_filenames[i]
            
            # Construct a more descriptive output filename
            base_name, ext = os.path.splitext(original_filename)
            output_filename = f"pca_{n_pca_components}d_layer_{layer_idx}_{base_name}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            
            torch.save(reduced_hs_tensor, output_path)
            print(f"  Saved: {output_path} with shape {reduced_hs_tensor.shape}")
            
        # Optionally, save the PCA model for this layer
        # pca_model_filename = f"pca_model_layer_{layer_idx}_components_{n_pca_components}.joblib"
        # pca_model_path = os.path.join(output_dir, pca_model_filename)
        # joblib.dump(fitted_pca_for_layer, pca_model_path)
        # print(f"  Saved PCA model for layer {layer_idx} to {pca_model_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()