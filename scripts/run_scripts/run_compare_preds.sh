#!/bin/bash
#SBATCH --partition=dev_cpu
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:20:00
#SBATCH --mem=16gb
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=compare_preds
#SBATCH --output=logs/compare_preds.out

source $HOME/master-thesis/.env/bin/activate

rsync -avhP $HOME/master-thesis/artifacts/xnli_en_val_multi_shot.json $TMPDIR/xnli_val.json

rsync -avhP $(ws_find ws_sascha)/predictions/custom_layer_predictions.jsonl $TMPDIR/custom_layer_predictions.jsonl

python $HOME/master-thesis/scripts/compare_preds.py $TMPDIR/xnli_val.json $TMPDIR/custom_layer_predictions.jsonl --key1 "text" --key2 "prediction" --show-mismatches

deactivate