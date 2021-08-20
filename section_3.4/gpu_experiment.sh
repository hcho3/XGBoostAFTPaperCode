#!/bin/bash
set -x
set -e

for fold in 1 2 3 4 5
do
  python gpu_experiment.py --tree_method hist --test_fold_id $fold
  python gpu_experiment.py --tree_method gpu_hist --test_fold_id $fold
done
