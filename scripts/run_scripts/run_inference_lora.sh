#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:30:00
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=infer_job
#SBATCH --output=$HOME/master-thesis/logs/infer_job.out


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

python $HOME/master-thesis/scripts/lora_inference.py \
    --base_model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --peft_model_path $HOME/master-thesis/run_outputs/models/1B_tied_lora \
    --test_file $HOME/master-thesis/artifacts/xnli_en_test_multi_shot.json \
    --output_file $HOME/master-thesis/run_outputs/test_inferences/1B_tied_predictions.jsonl \
    --batch_size 16 \
    --max_new_tokens 6 \
    --do_sample

deactivate
