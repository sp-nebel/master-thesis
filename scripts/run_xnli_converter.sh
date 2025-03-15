#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:20:00
#SBATCH --mem=8gb
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL

module load devel/python/3.12.3_gnu_13.3

source .danni_env/bin/activate

pip install jsonlines

python scripts/dataset_converter.py $1 $2 $3

deactivate
