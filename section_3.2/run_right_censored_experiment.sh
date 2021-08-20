#!/bin/bash
set -e

n_trials=2
for data_gen in coxph aft
do
  for method in XGBoostAFTNormal XGBoostAFTLogistic XGBoostAFTExtreme XGBoostCox SkSurvCoxLinear
  do
    for censoring_fraction in 0.0 0.2 0.5 0.8
    do
      mkdir -p "./${data_gen}-log"
      log_file="./${data_gen}-log/${method}-${censoring_fraction}.txt"
      echo "python -u run_right_censored_experiment.py --n_trials ${n_trials} --censoring_fraction ${censoring_fraction} --data_gen ${data_gen} --method ${method} > ${log_file}"
      python -u run_right_censored_experiment.py --n_trials ${n_trials} --censoring_fraction ${censoring_fraction} --data_gen ${data_gen} --method ${method} > "${log_file}"
    done
  done
done
