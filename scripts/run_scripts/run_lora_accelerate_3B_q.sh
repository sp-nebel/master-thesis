#!/bin/bash
#SBATCH --partition=gpu_a100_il
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:2
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=3B_tied_q
#SBATCH --output=logs/3B_tied_q.out


module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

source $HOME/master-thesis/.env/bin/activate

rsync -avhP $HOME/master-thesis/artifacts/xnli_en_train_no_s.json $TMPDIR/xnli_en_train.json

rsync -avhP $HOME/master-thesis/artifacts/xnli_en_val.json $TMPDIR/xnli_en_val.json

source $HOME/master-thesis/scripts/train_baseline_accelerate_3B_q.sh

rsync -avhP $TMPDIR/lora_model $HOME/master-thesis/run_outputs/models/3B_tied_q_only

deactivate