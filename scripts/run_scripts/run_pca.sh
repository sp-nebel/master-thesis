#!/bin/bash
#SBATCH --partition=dev_cpu
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:15:00
#SBATCH --mem=16gb
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=3B_pca_job
#SBATCH --output=$HOME/master-thesis/logs/3B_pca_states_job.out


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

python $HOME/master-thesis/scripts/pca_hs.py --input_dir $HOME/master-thesis/run_outputs/hidden_states/1B_base_hs --output_dir $HOME/master-thesis/run_outputs/pca_output/1B_base_pca

deactivate
