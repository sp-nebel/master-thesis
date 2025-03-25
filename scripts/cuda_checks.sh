#!/bin/bash
#SBATCH --partition=dev_gpu_4
#SBATCH --ntasks-per-node=30
#SBATCH --time=00:10:00
#SBATCH --mem=8gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=lora_job
#SBATCH --output=../lora_job.out

# Load modules
module load compiler/gnu/13.3
module load jupyter/ai/2024-11-29

source .env/bin/activate

# Check environment variables
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PATH: $PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

deactivate