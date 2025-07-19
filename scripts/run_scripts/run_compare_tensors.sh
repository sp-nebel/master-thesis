#!/bin/bash
#SBATCH --partition=dev_cpu
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:20:00
#SBATCH --mem=16gb
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=compare_tensors
#SBATCH --output=logs/compare_tensors.out

source $HOME/master-thesis/.env/bin/activate



python $HOME/master-thesis/scripts/compare_tensors.py $HOME/master-thesis/run_outputs/hidden_states/1B_hs_to_comp_1 $HOME/master-thesis/run_outputs/hidden_states/1B_hs_to_comp_2 


deactivate

