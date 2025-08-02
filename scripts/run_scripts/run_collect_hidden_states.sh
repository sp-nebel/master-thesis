#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:15:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=collect_hs
#SBATCH --output=logs/collect_hs.out


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

rsync -avhP $HOME/master-thesis/artifacts/xnli_en_test_multi_shot.json $TMPDIR/data.json

python $HOME/master-thesis/scripts/collect_last_hidden_states.py \
        --base_model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
        --test_file $TMPDIR/data.json \
        --output_file $TMPDIR/1B_hidden_states.pt \
        --trust_remote_code \
        --torch_dtype bfloat16 \
        --batch_size 16 \
        --max_input_length 512 \
        --layer_to_collect -1

rsync -avhP $TMPDIR/1B_hidden_states.pt $(ws_find ws_sascha)/hidden_states/1B_last_base_hidden_states_test_multi_shot.pt

python $HOME/master-thesis/scripts/collect_last_hidden_states.py \
        --base_model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
        --test_file $TMPDIR/data.json \
        --output_file $TMPDIR/3B_hidden_states.pt \
        --trust_remote_code \
        --torch_dtype bfloat16 \
        --batch_size 16 \
        --max_input_length 512 \
        --layer_to_collect -1

rsync -avhP $TMPDIR/3B_hidden_states.pt $(ws_find ws_sascha)/hidden_states/3B_last_base_hidden_states_test_multi_shot.pt

deactivate
