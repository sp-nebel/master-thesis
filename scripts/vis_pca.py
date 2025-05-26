import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load one of your PCA-reduced hidden state files
# Replace with the actual path to your .pt file
file_path = 'run_outputs/pca_output/3B_base/pca_3d_hidden_states_input_tokens_all_layers_0.pt'
hidden_states_3d = torch.load(file_path)
print(hidden_states_3d.shape)

# Ensure it's on the CPU and convert to NumPy for plotting
hidden_states_3d_np = hidden_states_3d.cpu().numpy()

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(hidden_states_3d_np[:, 0], hidden_states_3d_np[:, 1], hidden_states_3d_np[:, 2], s=10) # s is marker size

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title(f'3D PCA of Hidden States from {file_path.split("/")[-1]}')

# To save the figure:
plt.savefig('pca_visualization_layer_0.png')
print('Done')