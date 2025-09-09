#!/bin/bash
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:05:00
#SBATCH --mem=32gb
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=train_svd
#SBATCH --output=logs/train_svd_%j.out

module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

source $HOME/master-thesis/.env/bin/activate

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <input_file_1> <input_file_2> <layer> <output_dir>"
    exit 1
fi

INPUT_FILE_1=$1
INPUT_FILE_2=$2
LAYER=$3
OUTPUT_DIR=$4

rsync -avhP $INPUT_FILE_1 $TMPDIR/1B_hidden_states.pt

rsync -avhP $INPUT_FILE_2 $TMPDIR/3B_hidden_states.pt 

python $HOME/master-thesis/scripts/train_svds.py $TMPDIR/1B_hidden_states.pt $TMPDIR/3B_hidden_states.pt --layer $LAYER --output_dir $TMPDIR/mappings/

rsync -avhP $TMPDIR/mappings/ $OUTPUT_DIR/

deactivate