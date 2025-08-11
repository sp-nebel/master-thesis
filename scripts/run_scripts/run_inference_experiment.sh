#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:15:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=experiment_job
#SBATCH --output=logs/experiment_job.out


module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

source $HOME/master-thesis/.env/bin/activate

rsync -avhP $HOME/master-thesis/artifacts/xnli_en_val.json $TMPDIR/xnli_val.json

rsync -avhP $HOME/master-thesis/run_outputs/models/1B_tied_lora $TMPDIR/

rsync -avhP $HOME/master-thesis/run_outputs/proc_align/pre_q $TMPDIR/


python $HOME/master-thesis/scripts/inference_experiment.py \
    --base_model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --peft_model_path $TMPDIR/1B_tied_lora \
    --mapping_directory $TMPDIR/pre_q \
    --test_file $TMPDIR/xnli_val.json \
    --output_file $TMPDIR/experiment_q_proj_all_layers_val_preds.jsonl \
    --batch_size 16 \
    --max_new_tokens 6 \
    --do_sample

rsync -avhP $TMPDIR/experiment_q_proj_all_layers_val_preds.jsonl $(ws_find ws_sascha)/predictions/experiment_q_proj_all_layers_val_preds.jsonl


deactivate
