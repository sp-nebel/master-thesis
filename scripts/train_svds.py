import argparse
import torch



def main():
    # Load the torch matrices from the file paths
    small = torch.load(args.small_hidden_states)
    big = torch.load(args.big_hidden_states)
    
    print(f"Small matrix shape: {small.shape}")
    print(f"Big matrix shape: {big.shape}")

    U_b, S_b, V_b = torch.linalg.svd(big, full_matrices=False)
    U_s, S_s, V_s = torch.linalg.svd(small, full_matrices=False)

    U_b_t = U_b[:, :args.rank]
    V_b_t = V_b.T[:, :args.rank]
    U_s_t = U_s[:, :args.rank]
    V_s_t = V_s.T[:, :args.rank]

    P_U_up = torch.linalg.lstsq(U_s_t, U_b_t).solution
    P_V_up = torch.linalg.lstsq(V_s_t, V_b_t).solution

    P_U_down = torch.linalg.lstsq(U_b_t, U_s_t).solution
    P_V_down = torch.linalg.lstsq(V_b_t, V_s_t).solution

    aligned_U_b = U_b_t @ P_U_up
    aligned_V_b = V_b_t @ P_V_up

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process two file paths")
    parser.add_argument("small_hidden_states", help="Small hidden states file path")
    parser.add_argument("big_hidden_states", help="Big hidden states file path")
    parser.add_argument("--rank", "-r", type=int, default=320, help="Rank for SVD decomposition (default: 320)")

    args = parser.parse_args()

    # Your code here using args.small_hidden_states and args.big_hidden_states
    print(f"Small Hidden States: {args.small_hidden_states}")
    print(f"Big Hidden States: {args.big_hidden_states}")

    main()