#!/bin/bash
set -x
set -e

for dataset in ATAC_JV_adipose CTCF_TDH_ENCODE
do
  for hp in learning_rate max_depth min_child_weight reg_alpha reg_lambda aft_loss_distribution_scale
  do
    for dist in normal logistic extreme
    do
      for i in 1 2 3 4
      do
        python grid.py --dataset $dataset --distribution $dist --test_fold_id $i --nthread 4 --s3_bucket aft-experiment-logs --hyperparameters $hp
      done
    done
  done
done
