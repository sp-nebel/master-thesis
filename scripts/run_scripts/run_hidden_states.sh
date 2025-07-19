#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:15:00
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=hidden_job
#SBATCH --output=logs/hidden_states_job.out


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

python $HOME/master-thesis/scripts/inference_hs_in.py \
    --base_model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --test_file $HOME/master-thesis/artifacts/xnli_en_test_10.json \
    --output_file $HOME/master-thesis/run_outputs/1b_hs_to_comp \
    --hidden_states_dir $HOME/master-thesis/run_outputs/hidden_states/1B_hs_to_comp_2 \
    --batch_size 16 \
    --max_new_tokens 6 \
    --do_sample

deactivate
