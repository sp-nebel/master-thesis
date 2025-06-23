import os
import sys
import numpy as np
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.getcwd()) 

from scripts import mapping_nn

def compute_mapping_accuracy(x_pred_np, x_true_np, bsz=512):
    """
    Computes nearest neighbor accuracy for a 1-to-1 mapping in a memory-efficient way.
    """
    n_vectors = len(x_pred_np)
    acc = 0.0

    x_pred_np /= np.linalg.norm(x_pred_np, axis=1)[:, np.newaxis] + 1e-8
    x_true_np /= np.linalg.norm(x_true_np, axis=1)[:, np.newaxis] + 1e-8

    for i in range(0, n_vectors, bsz):
        print(f"Processing batch {i} of {n_vectors}")
        e = min(i + bsz, n_vectors)
        
        pred_batch = x_pred_np[i:e]

        scores = np.dot(x_true_np, pred_batch.T)

        pred_indices = scores.argmax(axis=0)

        for j in range(i, e):
            if pred_indices[j - i] == j:
                acc += 1.0
                
    return acc / n_vectors

def main(args):
    print("Loading model...")
    model = mapping_nn.HiddenStateAlignmentNet(2048, 3072, 1024, 512)
    cpu = torch.device('cpu')
    device = torch.device('cuda')
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    print("Model loaded.")

    print("Loading source...")
    source_tensor = torch.load(args.source_path, map_location=cpu).to(torch.float32)
    print("Loading target...")
    target_tensor = torch.load(args.target_path, map_location=cpu).to(torch.float32)
    print("Loaded source and target.")

    model.to(device)

    print(f"Source shape: {source_tensor.shape}")
    print(f"Target shape: {target_tensor.shape}")

    full_dataset = TensorDataset(source_tensor)
    dataloader = DataLoader(full_dataset, batch_size=256, shuffle=False, num_workers=0)

    print("Generating predictions...")
    predictions = []
    with torch.no_grad():
        for i, source_batch in enumerate(dataloader):
            print(f"Processing batch {i} of {len(dataloader)}")
            batch_input_on_device = source_batch[0].to(device) 
            predictions.append(model(batch_input_on_device).cpu())
    
    print("Predictions generated.")
    predicted_tensor = torch.cat(predictions, dim=0)
    predicted_np = predicted_tensor.numpy()
    target_np = target_tensor.cpu().numpy()

    print(f"Predicted shape: {predicted_tensor.shape}")
    print(f"Sample predicted values: {predicted_np[:5, :5]}")
    print(f"Sample target values: {target_np[:5, :5]}")

    print("Calculating nearest neighbor accuracy...")
    accuracy = compute_mapping_accuracy(predicted_np, target_np)
    
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--source_path")
    parser.add_argument("--target_path")

    args = parser.parse_args()
    main(args)