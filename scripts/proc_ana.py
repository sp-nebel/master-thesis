import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import procrustes
import numpy as np 

def plot_procrustes_aligned_trajectories(file1_path, file2_path, model1_name="Model 1 (Standardized)", model2_name="Model 2 (Aligned)"):
    """
    Loads two .pt files, performs Procrustes analysis, and plots the 
    aligned 3D trajectories.

    Args:
        file1_path (str): Path to the first .pt file.
        file2_path (str): Path to the second .pt file.
        model1_name (str): Name for the first model's legend.
        model2_name (str): Name for the second model's legend.
    """
    # Load tensors and convert to NumPy
    try:
        # Ensure tensors are loaded onto CPU and converted to NumPy
        tensor1 = torch.load(file1_path).cpu().numpy()
        tensor2 = torch.load(file2_path).cpu().numpy()
        print(f"Loaded {file1_path} with shape: {tensor1.shape}")
        print(f"Loaded {file2_path} with shape: {tensor2.shape}")
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Please ensure the paths are correct.")
        return
    except Exception as e:
        print(f"An error occurred loading tensors: {e}")
        return

    # --- Data Validation ---
    if tensor1.shape[1] != 3 or tensor2.shape[1] != 3:
        print("Error: Tensors must have a shape of (N, 3).")
        return
    if tensor1.shape[0] != tensor2.shape[0]:
        print("Error: Procrustes requires both sets to have the same number of points.")
        return

    # --- Perform Procrustes Analysis ---
    # mtx1: Standardized version of tensor1 (translated to origin, scaled to ||X||=1)
    # mtx2: tensor2 transformed (translated, scaled, rotated) to best fit mtx1
    # disparity: Sum of squared differences (M^2) between mtx1 and mtx2
    mtx1, mtx2, disparity = procrustes(tensor1, tensor2)
    print(f"Procrustes Disparity (M^2): {disparity:.4f}")

    # --- Create 3D Plot ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectories (using the Procrustes results)
    ax.plot(mtx1[:, 0], mtx1[:, 1], mtx1[:, 2], label=model1_name, marker='o', markersize=3, linestyle='-')
    ax.plot(mtx2[:, 0], mtx2[:, 1], mtx2[:, 2], label=model2_name, marker='x', markersize=3, linestyle='--')

    # Add start and end markers 
    ax.scatter(mtx1[0, 0], mtx1[0, 1], mtx1[0, 2], s=100, c='blue', marker='^', label=f'{model1_name} Start')
    ax.scatter(mtx1[-1, 0], mtx1[-1, 1], mtx1[-1, 2], s=100, c='blue', marker='s', label=f'{model1_name} End')
    ax.scatter(mtx2[0, 0], mtx2[0, 1], mtx2[0, 2], s=100, c='red', marker='^', label=f'{model2_name} Start')
    ax.scatter(mtx2[-1, 0], mtx2[-1, 1], mtx2[-1, 2], s=100, c='red', marker='s', label=f'{model2_name} End')

    # Set labels and title
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.set_title(f"Procrustes Aligned Trajectories (Disparity = {disparity:.4f})")
    ax.legend()
    ax.grid(True)
    
    # Try to set a more equal aspect ratio for 3D
    # Note: True 'equal' aspect ratio in matplotlib 3D can be tricky.
    # This is an attempt to make the scales visually similar.
    all_data = np.concatenate((mtx1, mtx2), axis=0)
    max_range = np.array([all_data[:,0].max()-all_data[:,0].min(), 
                          all_data[:,1].max()-all_data[:,1].min(), 
                          all_data[:,2].max()-all_data[:,2].min()]).max() / 2.0

    mid_x = (all_data[:,0].max()+all_data[:,0].min()) * 0.5
    mid_y = (all_data[:,1].max()+all_data[:,1].min()) * 0.5
    mid_z = (all_data[:,2].max()+all_data[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.savefig("run_outputs/procrustes_aligned_trajectories.png", dpi=300)

# --- Example Usage ---
# Make sure you have 'torch', 'matplotlib', and 'scipy' installed.
# You might need to create dummy files if you don't have them, e.g.:
# torch.save(torch.randn(145, 3), 'model1_seq1.pt')
# torch.save(torch.randn(145, 3) + 0.5 * torch.sin(torch.linspace(0, 10, 145)).unsqueeze(1), 'model2_seq1.pt') 

if __name__ == "__main__":
        plot_procrustes_aligned_trajectories('run_outputs/pca_output/1B_base/pca_3d_hidden_states_input_tokens_all_layers_0.pt', 'run_outputs/pca_output/3B_base/pca_3d_hidden_states_input_tokens_all_layers_0.pt')