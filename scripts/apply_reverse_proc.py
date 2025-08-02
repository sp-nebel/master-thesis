import argparse
import sys
import os
import numpy as np

import torch

from utils import top_knn_acc

def main(args):

    sys.path.append(os.getcwd()) 

    input_matrix = torch.load(args.input_path).to(torch.float32)
    proc_matrix = torch.load(args.proc_path).to(torch.float32)
    target_matrix = torch.load(args.target_path).to(torch.float32)

    input_matrix = input_matrix[:args.samples]
    target_matrix = target_matrix[:args.samples]   

    pred_matrix = torch.matmul(input_matrix, proc_matrix)

    pred_matrix = pred_matrix[:, :2048]  

    print(pred_matrix)

    mse_loss = torch.nn.MSELoss()

    mse_result = mse_loss(pred_matrix, target_matrix)

    print("MSE Result: ", mse_result)

    topk_accuracy, distances = top_knn_acc(args.k, pred_matrix.numpy(), target_matrix.numpy())

    print("NN Result: ", topk_accuracy)
    
    mean_distances = np.mean(distances, axis=0)
    std_distances = np.std(distances, axis=0)
    min_distances = np.min(distances, axis=0)
    max_distances = np.max(distances, axis=0)
    median_distances = np.median(distances, axis=0)
    
    print("\nDistance Statistics:")
    for k in range(args.k):
        print(f"Top-{k+1} distances - Mean: {mean_distances[k]:.4f}, Std: {std_distances[k]:.4f}, "
              f"Min: {min_distances[k]:.4f}, Max: {max_distances[k]:.4f}, Median: {median_distances[k]:.4f}")
    
    stats = {
        'accuracy': topk_accuracy,
        'mean_distances': mean_distances,
        'std_distances': std_distances,
        'min_distances': min_distances,
        'max_distances': max_distances,
        'median_distances': median_distances,
        'mse_result': mse_result.item()
    }
    
    if args.stats_path:
        np.savez(args.stats_path, **stats)
        print(f"Statistics saved to: {args.stats_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("target_path", type=str)
    parser.add_argument("proc_path", type=str)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--stats_path", type=str, help="Path to save statistics")
    args = parser.parse_args()
    main(args)