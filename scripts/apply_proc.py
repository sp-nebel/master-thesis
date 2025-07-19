import argparse
import sys
import os

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

    nn_result = top_knn_acc(args.k, pred_matrix.numpy(), target_matrix.numpy())

    print("NN Result: ", nn_result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("target_path", type=str)
    parser.add_argument("proc_path", type=str)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    main(args)