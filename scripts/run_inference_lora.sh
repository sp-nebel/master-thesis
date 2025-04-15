#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:10:00
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=infer_job
#SBATCH --output=./logs/infer_job.out


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

python scripts/lora_inference.py \
    --base_model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --test_file artifacts/xnli_en_test.json \
    --peft_model_path test_run_outputs/checkpoint-5000 \
    --output_file test_inferences/predictions.jsonl \
    --batch_size 16 \
    --max_new_tokens 4 \
    --do_sample

deactivate
