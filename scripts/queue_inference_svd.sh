#!/bin/bash

# This script queues the run_inference_experiment_svd.sh script
# with different permutations of Q_SCALE and V_SCALE values.

SCALES=$(seq 2 2 12)

for q_scale in $SCALES
do
    for v_scale in $SCALES
    do
        echo "Queuing job with Q_SCALE=$q_scale and V_SCALE=$v_scale"
        sbatch scripts/run_scripts/run_inference_experiment_svd.sh $q_scale $v_scale
    done
done

echo "All jobs have been queued."