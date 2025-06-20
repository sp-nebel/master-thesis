#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:10:00
#SBATCH --mem=8gb
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=xnli_converter
#SBATCH --output=$HOME/master-thesis/logs/xnli_converter.out

module load devel/python/3.12.3-gnu-14.2

source $HOME/master-thesis/.env/bin/activate

pip install jsonlines datasets

python $HOME/master-thesis/scripts/dataset_converter.py $1 $2 $3

deactivate
