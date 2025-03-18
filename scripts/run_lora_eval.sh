#!/bin/bash
#SBATCH --partition=dev_gpu_4
#SBATCH --ntasks-per-node=20
#SBATCH --time=00:20:00
#SBATCH --mem=8gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=lora_eval_job
#SBATCH --output=../lora_eval_job.out


module load compiler/gnu/13.3
module load devel/cuda/12.0
module load devel/python/3.12.3_gnu_13.3

source .danni_env/bin/activate

#pip install -e .
# pytorch
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120
# deepspeed
#pip install deepspeed
# other huggingface packags
#pip install datasets evaluate peft
# helper packages
#pip install scikit-learn hf_mtask_trainer 
# for evaluation
#pip install seqeval levenshtein

source scripts/eval_lora.sh

deactivate
