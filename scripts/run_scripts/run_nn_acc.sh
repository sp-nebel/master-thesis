#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:30:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=nn_acc
#SBATCH --output=logs/nn_val_128k_acc.out


module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

source $HOME/master-thesis/.env/bin/activate

rsync -avhP $(ws_find ws_sascha)/hidden_states/1B_base_hidden_states_val_128k.pt $TMPDIR/1B_hidden_states.pt

rsync -avhP $(ws_find ws_sascha)/hidden_states/3B_base_hidden_states_val_128k.pt $TMPDIR/3B_hidden_states.pt

python -u $HOME/master-thesis/scripts/nn_align_nn_acc.py \
        --model_path "$HOME/master-thesis/run_outputs/mapping_models/val_trained_linear_nn_post_q_128k.pth" \
        --source_path "$TMPDIR/1B_hidden_states.pt" \
        --target_path "$TMPDIR/3B_hidden_states.pt" \
        --stats_path "$TMPDIR/1B_to_3B_alignment_stats_10k.npz" \
        --num_examples 10000

rsync -avhP "$TMPDIR/1B_to_3B_alignment_stats_10k.npz" "$HOME/master-thesis/run_outputs/mapping_models/1B_to_3B_alignment_stats_10k.npz"


deactivate
