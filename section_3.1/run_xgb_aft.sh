#!/bin/bash

set -e
set -u
shopt -s failglob
set -o pipefail

n_trials=100
for dataset in ATAC_JV_adipose CTCF_TDH_ENCODE H3K27ac-H3K4me3_TDHAM_BP H3K27ac_TDH_some \
               H3K36me3_AM_immune H3K27me3_RL_cancer H3K27me3_TDH_some H3K36me3_TDH_ENCODE \
               H3K36me3_TDH_immune H3K36me3_TDH_other
do
    mkdir -p xgb-aft-log/${dataset}
    echo "python -u run_xgb_aft.py --data_source chipseq --dataset ${dataset} --n_trials ${n_trials} &> xgb-aft-log/${dataset}/log.txt"
    python -u run_xgb_aft.py --data_source chipseq --dataset ${dataset} --n_trials ${n_trials} &> xgb-aft-log/${dataset}/log.txt
done

for dataset in simulated.sin simulated.abs simulated.linear simulated.model.1 simulated.model.2 simulated.model.3
do
    mkdir -p xgb-aft-log/${dataset}
    echo "python -u run_xgb_aft.py --data_source simulated --dataset ${dataset} --n_trials ${n_trials} &> xgb-aft-log/${dataset}/log.txt"
    python -u run_xgb_aft.py --data_source simulated --dataset ${dataset} --n_trials ${n_trials} &> xgb-aft-log/${dataset}/log.txt
done
