#!/bin/bash
#SBATCH --partition=dev_cpu
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:05:00
#SBATCH --mem=16gb
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=compare_preds
#SBATCH --output=logs/acc_svds_dyn_r_all_layers.out

source $HOME/master-thesis/.env/bin/activate

rsync -avhP $HOME/master-thesis/artifacts/xnli_en_test_no_s.json $TMPDIR/xnli_test.json

rsync -avhP $HOME/master-thesis/run_outputs/predictions/experiment_dev_2_2_output.jsonl $TMPDIR/custom_layer_predictions.jsonl

python $HOME/master-thesis/scripts/compute_accuracy.py $TMPDIR/xnli_test.json $TMPDIR/custom_layer_predictions.jsonl --key1 "text" --key2 "prediction" --verbose

deactivate