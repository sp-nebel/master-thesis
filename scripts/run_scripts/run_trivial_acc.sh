#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:20:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=trivial_acc
#SBATCH --output=logs/trivial_acc.out


module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

source $HOME/master-thesis/.env/bin/activate

rsync -avhP $(ws_find ws_sascha)/hidden_states/1B_base_hidden_states_val_128k.pt $TMPDIR/1B_hidden_states.pt


python -u $HOME/master-thesis/scripts/nn_align_nn_acc.py \
        --model_path "$HOME/master-thesis/run_outputs/mapping_models/trivial_mapping_128k.pth" \
        --source_path "$TMPDIR/1B_hidden_states.pt" \
        --target_path "$TMPDIR/1B_hidden_states.pt" 


deactivate
