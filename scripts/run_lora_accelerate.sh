#!/bin/bash
#SBATCH --partition=gpu_h100 
#SBATCH --ntasks-per-node=40
#SBATCH --time=02:15:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:2
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=3B_tied
#SBATCH --output=./logs/3B_tied.out


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

source scripts/train_baseline_accelerate.sh

deactivate
