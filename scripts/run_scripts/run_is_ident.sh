#!/bin/bash
#SBATCH --partition=dev_cpu
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:10:00
#SBATCH --mem=8gb
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=is_ident
#SBATCH --output=logs/is_ident.out

module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

source $HOME/master-thesis/.env/bin/activate

# Copy matrices to temporary directory
rsync -avhP $HOME/master-thesis/run_outputs/proc_align/penult_proc_down_2M.pt $TMPDIR/matrix1.pt
rsync -avhP $HOME/master-thesis/run_outputs/proc_align/penult_proc_up_2M.pt $TMPDIR/matrix2.pt

# Run the identity check
python -u $HOME/master-thesis/scripts/is_ident.py $TMPDIR/matrix1.pt $TMPDIR/matrix2.pt --device cpu

deactivate