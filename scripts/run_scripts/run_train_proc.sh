#!/bin/bash
#SBATCH --partition=dev_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --mem=64gb
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=train_proc
#SBATCH --output=logs/train_proc_%j.out


module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

source $HOME/master-thesis/.env/bin/activate

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_file_1> <input_file_2> <output_filename>"
    exit 1
fi

INPUT_FILE_1=$1
INPUT_FILE_2=$2
OUTPUT_FILENAME=$3

rsync -avhP $INPUT_FILE_1 $TMPDIR/1B_hidden_states.pt

rsync -avhP $INPUT_FILE_2 $TMPDIR/3B_hidden_states.pt 

python $HOME/master-thesis/scripts/train_proc.py $TMPDIR/1B_hidden_states.pt $TMPDIR/3B_hidden_states.pt $TMPDIR/ --output_filename=${OUTPUT_FILENAME}_up.pt

python $HOME/master-thesis/scripts/train_proc.py $TMPDIR/3B_hidden_states.pt $TMPDIR/1B_hidden_states.pt $TMPDIR/ --output_filename=${OUTPUT_FILENAME}_down.pt

rsync -avhP $TMPDIR/${OUTPUT_FILENAME}_up.pt $HOME/master-thesis/run_outputs/proc_align/pre_q_orth/${OUTPUT_FILENAME}up.pt

rsync -avhP $TMPDIR/${OUTPUT_FILENAME}_down.pt $HOME/master-thesis/run_outputs/proc_align/pre_q_orth/${OUTPUT_FILENAME}down.pt

deactivate
