import utils
import os
import sys
import numpy as np
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.getcwd()) 

from scripts import train_nn

def main(args):
    print("Loading model...")
    model = train_nn.HiddenStateAlignmentNet(2048, 3072, 1024, 512)
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

    if args.num_examples:
        print(f"Using {args.num_examples} examples.")
        source_tensor = source_tensor[:args.num_examples]
        target_tensor = target_tensor[:args.num_examples]


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
    topk_accuracy, distances = utils.top_knn_acc(5, target_np, predicted_np)
    
    print(f"Accuracy: {topk_accuracy:.4f}")
    mean_distances = np.mean(distances, axis=0)
    std_distances = np.std(distances, axis=0)
    min_distances = np.min(distances, axis=0)
    max_distances = np.max(distances, axis=0)
    median_distances = np.median(distances, axis=0)
    
    print("\nDistance Statistics:")
    for k in range(5):
        print(f"Top-{k+1} distances - Mean: {mean_distances[k]:.4f}, Std: {std_distances[k]:.4f}, "
              f"Min: {min_distances[k]:.4f}, Max: {max_distances[k]:.4f}, Median: {median_distances[k]:.4f}")
    
    stats = {
        'accuracy': topk_accuracy,
        'mean_distances': mean_distances,
        'std_distances': std_distances,
        'min_distances': min_distances,
        'max_distances': max_distances,
        'median_distances': median_distances
    }
    

    np.savez(args.stats_path, **stats)
    print(f"Distance statistics saved to: {args.stats_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--source_path")
    parser.add_argument("--target_path")
    parser.add_argument("--stats_path")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of examples to use.")

    args = parser.parse_args()
    main(args)