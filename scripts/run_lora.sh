#!/bin/bash
#SBATCH --partition=dev_gpu_4
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:10:00
#SBATCH --mem=4gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=lora_job
#SBATCH --output=../lora_job.out


module load compiler/gnu/13.3
module load devel/cuda/12.0
module load devel/python/3.12.3_gnu_13.3

source .env/bin/activate

pip install -e .
# pytorch
pip install torch --index-url https://download.pytorch.org/whl/cu120
# deepspeed
pip install deepspeed
# other huggingface packags
pip install datasets evaluate peft
# helper packages
pip install scikit-learn hf_mtask_trainer 
# for evaluation
pip install seqeval levenshtein

source scripts/train_baseline.sh

deactivate
