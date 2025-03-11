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

module load devel/miniconda
module load devel/cuda/12.4
module load jupyter/ai
conda activate parrot_test_env
python3 setup.py
conda deactivate
