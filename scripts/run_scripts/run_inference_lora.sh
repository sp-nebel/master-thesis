#!/bin/bash
#SBATCH --partition=gpu_a100_short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:20:00
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=infer_job
#SBATCH --output=logs/3B_ootb_on_multi_shot.out


module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

source $HOME/master-thesis/.env/bin/activate

# rsync -avhP $HOME/master-thesis/run_outputs/models/1B_untied_v_q $TMPDIR/

rsync -avhP $HOME/master-thesis/artifacts/xnli_en_test_multi_shot.json $TMPDIR/xnli_test.json

python $HOME/master-thesis/scripts/lora_inference.py \
    --base_model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --test_file $TMPDIR/xnli_test.json \
    --output_file $TMPDIR/predictions.jsonl \
    --batch_size 16 \
    --max_new_tokens 6 \
    --do_sample

python $HOME/master-thesis/scripts/compare_preds.py $TMPDIR/xnli_test.json $TMPDIR/predictions.jsonl --key1 "text" --key2 "prediction" --show-mismatches


deactivate
