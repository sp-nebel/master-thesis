#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:30:00
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=trivial_mapping
#SBATCH --output=logs/trivial_mapping.out


module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

source $HOME/master-thesis/.env/bin/activate

pip install -e .
# pytorch
pip install torch --index-url https://download.pytorch.org/whl/cu128
# deepspeed
pip install deepspeed
# other huggingface packags
pip install datasets evaluate peft
# helper packages
pip install scikit-learn hf_mtask_trainer
# for evaluation
pip install seqeval levenshtein

rsync -avhP $(ws_find ws_sascha)/hidden_states/1B_base_hidden_states_128k.pt $TMPDIR/1B_hidden_states.pt

rsync -avhP $(ws_find ws_sascha)/hidden_states/3B_base_hidden_states_128k.pt $TMPDIR/3B_hidden_states.pt 


python $HOME/master-thesis/scripts/train_nn.py \
        --input_data_path "$TMPDIR/1B_hidden_states.pt" \
        --target_data_path "$TMPDIR/1B_hidden_states.pt" \
        --output_path "$HOME/master-thesis/run_outputs/mapping_models/trivial_mapping_128k.pth" \
        --output_dim 2048 \
        --num_workers 4


deactivate
