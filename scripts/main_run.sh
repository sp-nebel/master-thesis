#!/bin/bash
#SBATCH --partition=gpu_h100 
#SBATCH --ntasks-per-node=40
#SBATCH --time=01:00:00
#SBATCH --mem=64gb
#SBATCH --gres=gpu:4
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=main_run
#SBATCH --output=./logs/main_run.out


module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

source .env/bin/activate

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

source $1

deactivate
