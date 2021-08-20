import pandas as pd
import numpy as np
import os
import argparse
import json
import mmit
import time

from data_importer import data_import

from sklearn.model_selection import KFold
from mmit import MaxMarginIntervalTree
from mmit.pruning import min_cost_complexity_pruning
from mmit.model_selection import GridSearchCV
from mmit.metrics import mean_squared_error

current_dir = os.path.dirname(os.path.abspath(__file__))

def get_accuracy(pred, y_lower, y_higher):
    res = list(map(lambda x : x[0] >= x[1] and x[0] <= x[2], zip(pred, y_lower, y_higher)))
    accuracy = sum(res) / len(res) 
    return accuracy

def get_range_max_min(X, y, nature='margin'):
    if nature == 'margin':
        sorted_limits = y.flatten()
        sorted_limits = sorted_limits[~np.isinf(sorted_limits)]
        sorted_limits.sort()
        range_max = sorted_limits.max() - sorted_limits.min()
        range_min = np.diff(sorted_limits)
        range_min = range_min[range_min > 0].min()
    elif nature == 'min_sample_split':
        range_min = 2
        range_max = X.shape[0]
    return range_min, range_max

def get_margin_range(range_min, range_max, n_margin_values=10):
    margin = [0.] + np.logspace(np.log10(range_min), np.log10(range_max), n_margin_values).tolist()
    return margin

def get_min_sample_split_sample(range_min, range_max, n_min_samples_split_values=10):
    min_samples_split = np.logspace(np.log10(range_min), np.log10(range_max),
                                    n_min_samples_split_values).astype(np.uint).tolist()
    return min_samples_split

def get_train_valid_test_splits(folds, test_fold_id, inputs, labels):
    X = inputs[folds != test_fold_id]
    X_test = inputs[folds == test_fold_id]
    y = labels[folds != test_fold_id]
    y_test = labels[folds == test_fold_id]
    return X, X_test, y, y_test

def mmit_fit(X, y, seed):
    range_min, range_max = get_range_max_min(X, y, nature='margin')
    margin = get_margin_range(range_min,range_max, n_margin_values=10)
    range_min, range_max = get_range_max_min(X, y, nature='min_sample_split')
    min_samples_split = get_min_sample_split_sample(range_min, range_max,
                                                    n_min_samples_split_values=10)
    cv_protocol = KFold(n_splits=5, shuffle=True, random_state=seed)
    param_grid = {"margin": margin, "loss": ["linear_hinge"], "max_depth": [1, 2, 4, 6],
                  "min_samples_split": min_samples_split}
    estimator = MaxMarginIntervalTree()
    cv = GridSearchCV(estimator, param_grid, cv=cv_protocol, n_jobs=-1)
    cv.fit(X, y)
    return cv

def run_nested_cv(inputs, labels, folds, seed):
    fold_ids = np.unique(folds)
    accuracy = {}
    run_time = {}
    for fold_id in fold_ids:
        start = time.time() 
        X, X_test, y, y_test = get_train_valid_test_splits(folds, fold_id, inputs, labels)
        cv = mmit_fit(X,y,seed)
        y_pred = cv.predict(X_test)
        accuracy[str(fold_id)] = get_accuracy(y_pred, y_test[:,0], y_test[:,1])
        end = time.time()
        time_taken = end - start
        run_time[str(fold_id)] = time_taken
        print(f'Time elapsed = {time_taken}')
    return accuracy, run_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', required=True, choices=['chipseq', 'simulated'])
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    inputs, labels, folds = data_import(current_dir, args.data_source, args.dataset)
    inputs = inputs.values
    if args.data_source == 'simulated':
        labels = labels[['min.log.penalty', ' max.log.penalty']].values
    else:
        labels = labels[['min.log.lambda', 'max.log.lambda']].values
    folds = folds.values.reshape(-1, )

    result_folder = os.path.join(current_dir, 'mmit-log', args.dataset)
    accuracy, run_time = run_nested_cv(inputs, labels, folds, 42)
    acc_file = os.path.join(result_folder, "accuracy.json")
    run_file = os.path.join(result_folder, "run_time.json")
    with open(acc_file, "w") as f:
        json.dump(accuracy, f)
    with open(run_file, "w") as f:
        json.dump(run_time, f)

if __name__ == '__main__':
    main()
