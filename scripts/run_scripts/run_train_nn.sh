#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:30:00
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=linear_nn
#SBATCH --output=logs/linear_nn.out


module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

source $HOME/master-thesis/.env/bin/activate

rsync -avhP $(ws_find ws_sascha)/hidden_states/1B_with_hook/input_layernorm/scratch/slurm_tmpdir/job_1066702/1B_layer_15_self_attn_q_proj_post.pt $TMPDIR/1B_hidden_states.pt

rsync -avhP $(ws_find ws_sascha)/hidden_states/3B_with_hook/input_layernorm/scratch/slurm_tmpdir/job_1065634/3B_layer_27_self_attn_q_proj_post.pt $TMPDIR/3B_hidden_states.pt 


python $HOME/master-thesis/scripts/train_nn.py \
        --input_data_path "$TMPDIR/1B_hidden_states.pt" \
        --target_data_path "$TMPDIR/1B_hidden_states.pt" \
        --output_path "$HOME/master-thesis/run_outputs/mapping_models/linear_nn_post_q_128k.pth" \
        --input_dim 2048 \
        --output_dim 3072 \
        --num_workers 4


deactivate
