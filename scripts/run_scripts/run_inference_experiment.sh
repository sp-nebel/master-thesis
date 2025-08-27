#!/bin/bash
#SBATCH --partition=gpu_a100_short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=experiment_job
#SBATCH --output=logs/3B_vanilla_14_q_only.out


module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

source $HOME/master-thesis/.env/bin/activate

rsync -avhP $HOME/master-thesis/artifacts/xnli_en_test.json $TMPDIR/xnli_test.json

#rsync -avhP $HOME/master-thesis/run_outputs/models/3B_tied_lora $TMPDIR/

rsync -avhP $HOME/master-thesis/run_outputs/models/1B_tied_q_only $TMPDIR/

rsync -avhP $HOME/master-thesis/run_outputs/proc_align/q_mappings $TMPDIR/

#rsync -avhP $HOME/master-thesis/run_outputs/proc_align/v_mappings $TMPDIR/


python $HOME/master-thesis/scripts/inference_experiment.py \
    --base_model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --graft_lora_path $TMPDIR/1B_tied_q_only \
    --graft_layers 27 26 25 24 23 22 21 20 19 18 17 16 15 14 \
    --module_mappings "self_attn.q_proj:$TMPDIR/q_mappings" \
    --test_file $TMPDIR/xnli_test.json \
    --output_file $TMPDIR/experiment_output.jsonl \
    --batch_size 16 \
    --max_new_tokens 6 \
    --do_sample

python $HOME/master-thesis/scripts/compare_preds.py $TMPDIR/xnli_test.json $TMPDIR/experiment_output.jsonl --key1 "text" --key2 "prediction" --show-mismatches

deactivate
