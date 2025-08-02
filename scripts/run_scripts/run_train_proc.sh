#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=40
#SBATCH --time=01:15:00
#SBATCH --mem=256gb
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=train_proc
#SBATCH --output=logs/train_proc.out


module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

source $HOME/master-thesis/.env/bin/activate

rsync -avhP $(ws_find ws_sascha)/hidden_states/1B_penultimate_base_hidden_states_test_multi_shot.pt $TMPDIR/1B_hidden_states.pt

rsync -avhP $(ws_find ws_sascha)/hidden_states/3B_penultimate_base_hidden_states_test_multi_shot.pt $TMPDIR/3B_hidden_states.pt 

python $HOME/master-thesis/scripts/train_proc.py $TMPDIR/3B_hidden_states.pt $TMPDIR/1B_hidden_states.pt $TMPDIR/ --output_filename=penult_proc_up_2M.pt

rsync -avhP $TMPDIR/penult_proc_up_2M.pt $HOME/master-thesis/run_outputs/proc_align/penult_proc_down_2M.pt

deactivate
