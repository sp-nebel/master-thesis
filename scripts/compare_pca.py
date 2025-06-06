import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.spatial import procrustes # Added import

def load_pca_data_for_layer(pca_output_dir, layer_idx, file_pattern_template="pca_*d_layer_{layer_idx}_*.pt"):
    """
    Loads PCA data for a specific layer.
    Adjust file_pattern_template if your PCA output filenames are different.
    """
    pattern = os.path.join(pca_output_dir, file_pattern_template.format(layer_idx=layer_idx))
    files = glob.glob(pattern)
    if not files:
        print(f"Warning: No PCA files found for layer {layer_idx} in {pca_output_dir} with pattern {pattern}")
        return None
    
    all_data = []
    for f_path in files:
        try:
            data = torch.load(f_path, map_location='cpu')
            all_data.append(data)
        except Exception as e:
            print(f"Error loading {f_path}: {e}")
            continue
    
    if not all_data:
        return None
    
    # Concatenate if multiple files were found for the same layer (e.g. from multiple input sequences)
    return torch.cat(all_data, dim=0).numpy()

def compare_pca_outputs_relative_depth(
    model1_pca_dir, model2_pca_dir, 
    num_layers_model1, num_layers_model2, 
    n_pca_components, output_plot_dir,
    model1_name="Model1", model2_name="Model2"
):
    if not os.path.exists(output_plot_dir):
        os.makedirs(output_plot_dir)
        print(f"Created plot directory: {output_plot_dir}")

    # Handle cases where a model might have only 1 layer to avoid division by zero
    denom_m1 = (num_layers_model1 - 1) if num_layers_model1 > 1 else 1
    denom_m2 = (num_layers_model2 - 1) if num_layers_model2 > 1 else 1

    for layer_idx_m1 in range(num_layers_model1):
        relative_depth_m1 = layer_idx_m1 / denom_m1 if num_layers_model1 > 1 else 0.5 # Treat single layer as mid-depth

        # Find corresponding layer in model 2
        target_layer_idx_m2_float = relative_depth_m1 * denom_m2
        layer_idx_m2 = int(round(target_layer_idx_m2_float))
        # Ensure it's within bounds of model2 layers
        layer_idx_m2 = max(0, min(layer_idx_m2, num_layers_model2 - 1))

        print(f"\n--- Comparing {model1_name} Layer {layer_idx_m1} (RelDepth: {relative_depth_m1:.2f}) "
              f"with {model2_name} Layer {layer_idx_m2} (TargetRelDepth: {target_layer_idx_m2_float/denom_m2:.2f}) ---")
        
        data_m1_orig = load_pca_data_for_layer(model1_pca_dir, layer_idx_m1)
        data_m2_orig = load_pca_data_for_layer(model2_pca_dir, layer_idx_m2)

        if data_m1_orig is None or data_m2_orig is None:
            print(f"Skipping comparison due to missing data for M1 L{layer_idx_m1} or M2 L{layer_idx_m2}.")
            continue
        
        if data_m1_orig.shape[1] != n_pca_components or data_m2_orig.shape[1] != n_pca_components:
            print(f"Warning: Data has unexpected number of components. "
                  f"{model1_name} L{layer_idx_m1}: {data_m1_orig.shape}, "
                  f"{model2_name} L{layer_idx_m2}: {data_m2_orig.shape}. Skipping.")
            continue

        # --- Procrustes Transformation ---
        # Procrustes requires the same number of points (samples).
        # We'll use the minimum number of samples from the two datasets.
        n_samples = min(data_m1_orig.shape[0], data_m2_orig.shape[0])
        
        if n_samples == 0:
            print(f"Warning: No samples to compare for M1 L{layer_idx_m1} and M2 L{layer_idx_m2} (n_samples=0). Skipping.")
            continue
        
        if data_m1_orig.shape[0] != data_m2_orig.shape[0]:
            print(f"Info: Different number of samples for Procrustes: "
                  f"{model1_name} L{layer_idx_m1} has {data_m1_orig.shape[0]}, "
                  f"{model2_name} L{layer_idx_m2} has {data_m2_orig.shape[0]}. "
                  f"Using first {n_samples} samples from each for Procrustes alignment.")

        data_m1_for_procrustes = data_m1_orig[:n_samples, :]
        data_m2_for_procrustes = data_m2_orig[:n_samples, :]

        try:
            # mtx1_aligned is standardized data_m1_for_procrustes
            # mtx2_aligned is standardized, rotated, and scaled data_m2_for_procrustes to best match mtx1_aligned
            # disparity is the sum of squared differences between mtx1_aligned and mtx2_aligned
            mtx1_aligned, mtx2_aligned, disparity = procrustes(data_m1_for_procrustes, data_m2_for_procrustes)
            print(f"  Procrustes analysis: disparity = {disparity:.4f} (using {n_samples} samples)")
            
            # Use the aligned matrices for comparison
            data_m1 = data_m1_orig
            data_m2 = data_m2_orig
            procrustes_applied = False

        except Exception as e: # Catching a broader range of exceptions like ValueError or LinAlgError
            print(f"Error during Procrustes analysis for M1 L{layer_idx_m1} and M2 L{layer_idx_m2}: {e}")
            print("  Skipping comparison for this layer pair due to Procrustes failure.")
            continue # Skip to the next layer_idx_m1

        # --- Visual Comparison ---
        N_COLS = 3
        # Determine the number of rows for the subplot grid based on your existing figure height logic
        # This logic correctly anticipates 1 row for 1 or 2 PCA components, and 2 rows for 3 components,
        # assuming 3 plots per row (1 scatter + 2 KDEs for n_pca_components=2; 3 scatters + 3 KDEs for n_pca_components=3).
        num_rows_fig = max(1, n_pca_components // 2 + n_pca_components % 2)
        if n_pca_components == 1: # Special case: 1 KDE plot
            num_rows_fig = 1
        elif n_pca_components == 2: # 1 scatter, 2 KDEs = 3 plots
            num_rows_fig = 1
        elif n_pca_components >= 3: # 3 scatters, n_pca_components KDEs. E.g., 3 components = 6 plots
            total_plots = 3 + n_pca_components 
            num_rows_fig = (total_plots + N_COLS - 1) // N_COLS


        plt.figure(figsize=(18, 5 * num_rows_fig)) # Adjust figure size

        plot_idx = 1
        # 1. Scatter Plot (if 2D or 3D)
        if n_pca_components >= 2:
            plt.subplot(num_rows_fig, N_COLS, plot_idx) # Use consistent row/col numbers
            plt.scatter(data_m1[:, 0], data_m1[:, 1], alpha=0.5, label=f'{model1_name} L{layer_idx_m1} (PC1vPC2)', s=10)
            plt.scatter(data_m2[:, 0], data_m2[:, 1], alpha=0.5, label=f'{model2_name} L{layer_idx_m2} (PC1vPC2)', s=10)
            plt.xlabel("PC1 (Procrustes Aligned)")
            plt.ylabel("PC2 (Procrustes Aligned)")
            plt.title(f"L{layer_idx_m1}({model1_name[0]})vL{layer_idx_m2}({model2_name[0]}) PC1vPC2 (Procrustes)")
            plt.legend()
            plt.grid(True)
            plot_idx +=1
        
        # Add another scatter for PC1 vs PC3 if n_pca_components >= 3
        if n_pca_components >= 3:
            plt.subplot(num_rows_fig, N_COLS, plot_idx) # Use consistent row/col numbers
            plt.scatter(data_m1[:, 0], data_m1[:, 2], alpha=0.5, label=f'{model1_name} L{layer_idx_m1} (PC1vPC3)', s=10)
            plt.scatter(data_m2[:, 0], data_m2[:, 2], alpha=0.5, label=f'{model2_name} L{layer_idx_m2} (PC1vPC3)', s=10)
            plt.xlabel("PC1")
            plt.ylabel("PC3")
            plt.title(f"L{layer_idx_m1}({model1_name[0]})vL{layer_idx_m2}({model2_name[0]}) PC1vPC3")
            plt.legend()
            plt.grid(True)
            plot_idx +=1

            plt.subplot(num_rows_fig, N_COLS, plot_idx) # Use consistent row/col numbers
            plt.scatter(data_m1[:, 1], data_m1[:, 2], alpha=0.5, label=f'{model1_name} L{layer_idx_m1} (PC2vPC3)', s=10)
            plt.scatter(data_m2[:, 1], data_m2[:, 2], alpha=0.5, label=f'{model2_name} L{layer_idx_m2} (PC2vPC3)', s=10)
            plt.xlabel("PC2")
            plt.ylabel("PC3")
            plt.title(f"L{layer_idx_m1}({model1_name[0]})vL{layer_idx_m2}({model2_name[0]}) PC2vPC3")
            plt.legend()
            plt.grid(True)
            plot_idx +=1

        # 2. Histograms/KDEs for each component
        for pc_idx in range(n_pca_components):
            # Ensure we don't try to create more subplots than available
            if plot_idx > num_rows_fig * N_COLS:
                print(f"Warning: plot_idx ({plot_idx}) exceeds maximum number of subplots ({num_rows_fig * N_COLS}). Skipping further KDE plots.")
                break
            
            plt.subplot(num_rows_fig, N_COLS, plot_idx) # Use consistent row/col numbers and current plot_idx
            sns.kdeplot(data_m1[:, pc_idx], label=f'{model1_name} L{layer_idx_m1}-PC{pc_idx+1}', fill=True, alpha=0.5)
            sns.kdeplot(data_m2[:, pc_idx], label=f'{model2_name} L{layer_idx_m2}-PC{pc_idx+1}', fill=True, alpha=0.5)
            plt.xlabel(f"PC{pc_idx+1} Value")
            plt.ylabel("Density")
            plt.title(f"L{layer_idx_m1}({model1_name[0]})vL{layer_idx_m2}({model2_name[0]}) KDE PC{pc_idx+1}")
            plt.legend()
            plt.grid(True)
            plot_idx +=1


            # --- Quantitative Comparison (Example) ---
            wd = wasserstein_distance(data_m1[:, pc_idx], data_m2[:, pc_idx])
            ks_stat, ks_p_value = ks_2samp(data_m1[:, pc_idx], data_m2[:, pc_idx])
            print(f"  PC{pc_idx+1}: Wasserstein Dist = {wd:.4f}, KS p-value = {ks_p_value:.4f} (Stat={ks_stat:.4f})")

        plt.tight_layout(pad=1.0)
        plot_filename = os.path.join(output_plot_dir, f"M1_L{layer_idx_m1}_vs_M2_L{layer_idx_m2}_comparison.png")
        plt.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}")
        plt.close()


if __name__ == "__main__":
    # --- Configuration ---
    N_PCA_COMPONENTS = 3 # Should match what you used in pca_hs.py

    # Example 1: Small Base (16L) vs. Small Tuned (16L) - Relative depth not strictly needed but function handles it
    # print("\n=== Comparing Small Base vs Small Tuned (Same Architecture) ===")
    # small_base_pca_dir = "run_outputs/pca_output/1B_base_pca" 
    # small_tuned_pca_dir = "run_outputs/pca_output/1B_lora_pca" 
    # small_comp_plot_dir = "run_outputs/pca_comparisons/small_base_vs_tuned_relative/"
    # NUM_LAYERS_SMALL = 16
    # compare_pca_outputs_relative_depth(
    #     small_base_pca_dir, small_tuned_pca_dir,
    #     NUM_LAYERS_SMALL, NUM_LAYERS_SMALL, # Same number of layers
    #     N_PCA_COMPONENTS, small_comp_plot_dir,
    #     model1_name="SmallBase", model2_name="SmallTuned"
    # )

    # Example 2: Small Base (16L) vs. Big Base (28L) - Using relative depth
    print("\n=== Comparing Small Model vs Big Model (Different Architectures) ===")
    small_pca_dir = "run_outputs/pca_output/1B_base_pca" 
    big_pca_dir = "run_outputs/pca_output/3B_base_pca" # Assuming this exists
    small_vs_big_plot_dir = "run_outputs/pca_comparisons/small_base_vs_big_base_relative/"
    NUM_LAYERS_SMALL = 16
    NUM_LAYERS_BIG = 28 # For your big model

    # Ensure the big model PCA directory exists
    if not os.path.exists(big_pca_dir):
        print(f"ERROR: PCA directory for the big model not found: {big_pca_dir}")
        print("Please generate PCA outputs for the big model first using scripts/pca_hs.py.")
        print(f"Example: python scripts/pca_hs.py --input_dir run_outputs/hidden_states/3B_base_hs --output_dir {big_pca_dir} --n_pca_components {N_PCA_COMPONENTS}")
    else:
        compare_pca_outputs_relative_depth(
            small_pca_dir, big_pca_dir,
            NUM_LAYERS_SMALL, NUM_LAYERS_BIG,
            N_PCA_COMPONENTS, small_vs_big_plot_dir,
            model1_name="SmallBase", model2_name="BigBase"
        )

    # You can add more comparisons:
    # e.g., Small Tuned vs. Big Tuned
    # small_tuned_pca_dir = "run_outputs/pca_output/1B_lora_pca"
    # big_tuned_pca_dir = "run_outputs/pca_output/3B_lora_pca" # Assuming this exists
    # small_tuned_vs_big_tuned_plot_dir = "run_outputs/pca_comparisons/small_tuned_vs_big_tuned_relative/"
    # if not os.path.exists(big_tuned_pca_dir):
    #     print(f"ERROR: PCA directory for the big tuned model not found: {big_tuned_pca_dir}")
    # else:
    #     compare_pca_outputs_relative_depth(
    #         small_tuned_pca_dir, big_tuned_pca_dir,
    #         NUM_LAYERS_SMALL, NUM_LAYERS_BIG,
    #         N_PCA_COMPONENTS, small_tuned_vs_big_tuned_plot_dir,
    #         model1_name="SmallTuned", model2_name="BigTuned"
    #     )