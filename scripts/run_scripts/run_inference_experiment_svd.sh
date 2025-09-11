#!/bin/bash
#SBATCH --partition=gpu_a100_short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:20:00
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=experiment_job
#SBATCH --output=logs/experiment_r_dyn.out

Q_SCALE=$1
V_SCALE=$2

if [ -z "$Q_SCALE" ] || [ -z "$V_SCALE" ]; then
    echo "Usage: Pass Q_SCALE and V_SCALE as arguments."
    echo "Example: sbatch $0 0.5 0.5"
    exit 1
fi

module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

source $HOME/master-thesis/.env/bin/activate

rsync -avhP $HOME/master-thesis/artifacts/xnli_en_test_no_s.json $TMPDIR/xnli_test.json

#rsync -avhP $HOME/master-thesis/run_outputs/models/3B_tied_lora $TMPDIR/

rsync -avhP $HOME/master-thesis/run_outputs/models/1B_tied_vq $TMPDIR/

rsync -avhP $HOME/master-thesis/run_outputs/svds/q_mappings_r_dyn/ $TMPDIR/q_mappings

rsync -avhP $HOME/master-thesis/run_outputs/svds/v_mappings_r_dyn/ $TMPDIR/v_mappings


python $HOME/master-thesis/scripts/inference_experiment_svd.py \
    --base_model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --graft_lora_path $TMPDIR/1B_tied_vq \
    --module_mappings self_attn.q_proj:$TMPDIR/q_mappings self_attn.v_proj:$TMPDIR/v_mappings \
    --graft_layers -1 \
    --q_proj_scaling $Q_SCALE \
    --v_proj_scaling $V_SCALE \
    --test_file $TMPDIR/xnli_test.json \
    --output_file $TMPDIR/experiment_output.jsonl \
    --batch_size 5 \
    --max_new_tokens 6 \
    --torch_dtype float32 \
    --do_sample

rsync -avhP $TMPDIR/experiment_output.jsonl $HOME/master-thesis/run_outputs/predictions/experiment_svd_dyn_${Q_SCALE}_${V_SCALE}_output.jsonl

deactivate
