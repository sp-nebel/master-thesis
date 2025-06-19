#!/bin/bash
#SBATCH --partition=dev_cpu
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:10:00
#SBATCH --mem=16gb
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=proc_align
#SBATCH --output=logs/proc_align.out


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

pip install matplotlib seaborn

pip install qc-procrustes

rsync -avhP $(ws_find ws_sascha)/hidden_states/1B_base_hidden_states_128k.pt $TMPDIR/1B_hidden_states.pt

rsync -avhP $(ws_find ws_sascha)/hidden_states/3B_base_hidden_states_128k.pt $TMPDIR/3B_hidden_states.pt 

python $HOME/master-thesis/scripts/proc_align.py $TMPDIR/1B_hidden_states.pt $TMPDIR/3B_hidden_states.pt $TMPDIR/procrustes_rotation_matrix.pt

rsync -avhP $TMPDIR/procrustes_rotation_matrix.pt $HOME/master-thesis/run_outputs/proc_align/procrustes_rotation_matrix.pt

deactivate
