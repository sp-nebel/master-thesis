import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_trajectories(file1_path, file2_path, model1_name="Model 1", model2_name="Model 2"):
    """
    Loads two .pt files containing (N, 3) tensors and plots them 
    as 3D trajectories.

    Args:
        file1_path (str): Path to the first .pt file.
        file2_path (str): Path to the second .pt file.
        model1_name (str): Name for the first model's legend.
        model2_name (str): Name for the second model's legend.
    """
    # Load tensors
    try:
        tensor1 = torch.load(file1_path)
        tensor2 = torch.load(file2_path)
        print(f"Loaded {file1_path} with shape: {tensor1.shape}")
        print(f"Loaded {file2_path} with shape: {tensor2.shape}")
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Please ensure the paths are correct.")
        return
    except Exception as e:
        print(f"An error occurred loading tensors: {e}")
        return

    # Ensure tensors are (N, 3)
    if tensor1.shape[1] != 3 or tensor2.shape[1] != 3:
        print("Error: Tensors must have a shape of (N, 3).")
        return

    # Convert to NumPy for plotting
    data1 = tensor1.cpu().numpy()
    data2 = tensor2.cpu().numpy()

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectories
    ax.plot(data1[:, 0], data1[:, 1], data1[:, 2], label=model1_name, marker='o', markersize=3, linestyle='-')
    ax.plot(data2[:, 0], data2[:, 1], data2[:, 2], label=model2_name, marker='x', markersize=3, linestyle='--')

    # Add start and end markers for clarity
    ax.scatter(data1[0, 0], data1[0, 1], data1[0, 2], s=100, c='blue', marker='^', label=f'{model1_name} Start')
    ax.scatter(data1[-1, 0], data1[-1, 1], data1[-1, 2], s=100, c='blue', marker='s', label=f'{model1_name} End')
    ax.scatter(data2[0, 0], data2[0, 1], data2[0, 2], s=100, c='red', marker='^', label=f'{model2_name} Start')
    ax.scatter(data2[-1, 0], data2[-1, 1], data2[-1, 2], s=100, c='red', marker='s', label=f'{model2_name} End')

    # Set labels and title
    ax.set_xlabel("PCA Dimension 1")
    ax.set_ylabel("PCA Dimension 2")
    ax.set_zlabel("PCA Dimension 3")
    ax.set_title("Comparison of LLM Hidden State Trajectories (3D PCA)")
    ax.legend()
    ax.grid(True)

    plt.savefig("run_outputs/3d_trajectories_comparison.png", dpi=300)

# --- Example Usage ---
# Ensure you have files named 'model1_seq1.pt' and 'model2_seq1.pt' 
# or change the paths accordingly.
# As an example, let's create dummy files:
# torch.save(torch.randn(145, 3), 'model1_seq1.pt')
# torch.save(torch.randn(145, 3) + 0.5, 'model2_seq1.pt') 

if __name__ == "__main__":
    plot_3d_trajectories('run_outputs/pca_output/1B_base/pca_3d_hidden_states_input_tokens_all_layers_0.pt', 'run_outputs/pca_output/3B_base/pca_3d_hidden_states_input_tokens_all_layers_0.pt')