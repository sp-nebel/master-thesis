import argparse
import sys
import os

import torch

from nn_align_nn_acc import compute_mapping_accuracy

def main(args):

    sys.path.append(os.getcwd()) 

    input_matrix = torch.load(args.input_path).to(torch.float32)
    proc_matrix = torch.load(args.proc_path).to(torch.float32)
    target_matrix = torch.load(args.target_path).to(torch.float32)

    input_matrix = torch.nn.functional.pad(input_matrix, (0, target_matrix.shape[1] - input_matrix.shape[1]))


    pred_matrix = torch.matmul(input_matrix, proc_matrix)

    mse_loss = torch.nn.MSELoss()

    mse_result = mse_loss(pred_matrix, target_matrix)

    print("MSE Result: ", mse_result)

    nn_result = compute_mapping_accuracy(pred_matrix.numpy(), target_matrix.numpy())

    print("NN Result: ", nn_result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("target_path", type=str)
    parser.add_argument("proc_path", type=str)
    args = parser.parse_args()
    main(args)