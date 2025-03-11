#!/bin/bash
#SBATCH --partition=dev_gpu_4
#SBATCH --ntasks-per-node=20
#SBATCH --time=00:20:00
#SBATCH --mem=8gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=danni_job
#SBATCH --output=../danni_job.out

module load devel/cuda
module load devel/cudnn
module load devel/miniconda
conda activate parrot_test_env
pip install -e .
# pytorch
pip install torch torchvision torchaudio
# deepspeed
pip install deepspeed
# other huggingface packags
pip install datasets evaluate peft
# helper packages
pip install scikit-learn hf_mtask_trainer 
# for evaluation
pip install seqeval levenshtein

source scripts/train_baseline.sh
conda deactivate
