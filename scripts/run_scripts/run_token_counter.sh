#!/bin/bash
#SBATCH --partition=dev_cpu
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:30:00
#SBATCH --mem=16gb
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=counter_job
#SBATCH --output=logs/counter_job.out

source $HOME/master-thesis/.env/bin/activate

python $HOME/master-thesis/scripts/token_counter.py \
  "$HOME/master-thesis/artifacts/xnli_en_test.json" \
  "meta-llama/Llama-3.2-1B-Instruct" \
  "prefix" \
  --token_threshold 128000 \
  --output_file "$TMPDIR/output.json"

cp $TMPDIR/output.json $HOME/master-thesis/artifacts/xnli_en_test_128k.json

deactivate