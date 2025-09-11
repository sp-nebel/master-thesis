#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=experiment_job
#SBATCH --output=logs/experiment_test_multi_shot_with_s_1.out


module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

source $HOME/master-thesis/.env/bin/activate

rsync -avhP $HOME/master-thesis/artifacts/xnli_en_test_no_s.json $TMPDIR/xnli_test.json

#rsync -avhP $HOME/master-thesis/run_outputs/models/3B_tied_v_q $TMPDIR/

rsync -avhP $HOME/master-thesis/run_outputs/models/1B_tied_q_only $TMPDIR/

rsync -avhP $HOME/master-thesis/run_outputs/proc_align/q_mappings $TMPDIR/

#rsync -avhP $HOME/master-thesis/run_outputs/proc_align/v_mappings $TMPDIR/


python $HOME/master-thesis/scripts/inference_experiment.py \
    --base_model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --graft_lora_path $TMPDIR/1B_tied_q_only \
    --module_mappings self_attn.q_proj:$TMPDIR/q_mappings \
    --graft_layers 27 \
    --test_file $TMPDIR/xnli_test.json \
    --output_file $TMPDIR/experiment_output.jsonl \
    --batch_size 16 \
    --max_new_tokens 6 \
    --torch_dtype float32 \
    --do_sample \

rsync -avhP $TMPDIR/experiment_output.jsonl $HOME/master-thesis/run_outputs/predictions/experiment_test_multi_shot_with_s_1_output.jsonl

deactivate
