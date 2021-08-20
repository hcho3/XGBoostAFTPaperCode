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
        python comb.py 3 learning_rate max_depth min_child_weight reg_alpha reg_lambda aft_loss_distribution_scale | while read params
        do
          log_file=$(python get_log_name.py --run grid --dataset $dataset --distribution $dist --test_fold_id $i --nthread 4 --s3_bucket aft-experiment-logs --hyperparameters ${params})
          if [ -f "$log_file" ]
          then
            echo "Experiment $log_file already occured. Skipping"
          else
            python grid.py --dataset $dataset --distribution $dist --test_fold_id $i --nthread 12 --s3_bucket aft-experiment-logs --hyperparameters ${params}
          fi
        done
      done
    done
  done
done
