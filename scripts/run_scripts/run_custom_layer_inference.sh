#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:30:00
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=infer_job
#SBATCH --output=logs/custom_layer_infer_job.out


module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

source $HOME/master-thesis/.env/bin/activate

rsync -avhP $HOME/master-thesis/artifacts/xnli_en_val_multi_shot.json $TMPDIR/xnli_val.json

rsync -avhP $HOME/master-thesis/run_outputs/proc_align/reverse_procrustes_128k.pt $TMPDIR/reverse_procrustes_128k.pt

rsync -avhP $HOME/master-thesis/run_outputs/proc_align/procrustes_rotation_matrix.pt $TMPDIR/procrustes_rotation_matrix.pt


python $HOME/master-thesis/scripts/inference_custom_layer.py \
    --base_model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --down_mapping $TMPDIR/reverse_procrustes_128k.pt\
    --up_mapping $TMPDIR/procrustes_rotation_matrix.pt \
    --test_file $TMPDIR/xnli_val.json \
    --output_file $TMPDIR/custom_layer_predictions.jsonl \
    --batch_size 16 \
    --max_new_tokens 6 \
    --do_sample

rsync -avhP $TMPDIR/custom_layer_predictions.jsonl $(ws_find ws_sascha)/predictions/custom_layer_predictions.jsonl


deactivate
