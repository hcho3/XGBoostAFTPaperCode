import pandas as pd
import numpy as np
import xgboost as xgb
import time
import argparse
from multiprocessing import cpu_count

def accuracy(predt, dmat):
    y_lower = dmat.get_float_info('label_lower_bound')
    y_upper = dmat.get_float_info('label_upper_bound')
    acc = np.sum((predt >= y_lower) & (predt <= y_upper)) / len(predt)
    return acc

def run(tree_method, test_fold_id):
    base_params = {'objective': 'survival:aft',
                   'tree_method': tree_method,
                   'eval_metric': 'aft-nloglik',
                   'aft_loss_distribution': 'logistic'}
    if tree_method == 'hist':
        nthread = cpu_count() // 2
        base_params['nthread'] = nthread
        print(f'Using {nthread} threads')

    params = {}
    params[1] = {'learning_rate': 0.03143083719027954,
                 'max_depth': 6,
                 'min_child_weight': 0.7448027924644145,
                 'reg_alpha': 0.0009011995836009879,
                 'reg_lambda': 0.01851315290680719,
                 'aft_loss_distribution_scale': 1.406196348388404}
    params[2] = {'learning_rate': 0.012264342348819082,
                 'max_depth': 10,
                 'min_child_weight': 0.17075741376396836,
                 'reg_alpha': 10.026998353640105,
                 'reg_lambda': 0.00013000719512276063,
                 'aft_loss_distribution_scale': 3.463668779783766}
    params[3] =  {'learning_rate': 0.02043911447646343,
                  'max_depth': 4,
                  'min_child_weight': 0.24465310481252453,
                  'reg_alpha': 0.0002084055665237361,
                  'reg_lambda': 0.016202147752939614,
                  'aft_loss_distribution_scale': 1.1352374079345373}
    params[4] = {'learning_rate': 0.009697307726760454,
                 'max_depth': 4,
                 'min_child_weight': 0.16120791765370118,
                 'reg_alpha': 0.28749837817464324,
                 'reg_lambda': 0.0011782020578892955,
                 'aft_loss_distribution_scale': 2.3285133258250315}
    params[5] = {'learning_rate': 0.017107249461804603,
                 'max_depth': 10,
                 'min_child_weight': 0.17354844141748269,
                 'reg_alpha': 0.0033679345541758536,
                 'reg_lambda': 0.0049241661330130864,
                 'aft_loss_distribution_scale': 4.133663227935156}

    nround = {1: 52, 2: 149, 3: 53, 4: 81, 5: 83}

    filename = f'https://raw.githubusercontent.com/avinashbarnwal/aftXgboostPaper/master/data/simulate/simulated.model.3/'
    inputFileName = filename + 'features.csv'
    labelFileName = filename + 'targets.csv'
    foldsFileName = filename + 'folds.csv'
    inputs        = pd.read_csv(inputFileName)
    labels        = pd.read_csv(labelFileName)
    folds         = pd.read_csv(foldsFileName)
    # pre-processing: change label scale
    labels['min.lambda'] = labels['min.log.penalty'].apply(lambda x: np.exp(float(x)))
    labels['max.lambda'] = labels[' max.log.penalty'].apply(lambda x: np.exp(float(x)))

    ntile = 100000

    X            = np.tile(inputs[folds['fold'] != test_fold_id].values, (ntile, 1))
    X_test       = np.tile(inputs[folds['fold'] == test_fold_id].values, (ntile, 1))
    y_label      = labels[folds['fold'] != test_fold_id]
    y_label_test = labels[folds['fold'] == test_fold_id]
    print((X.shape, X_test.shape))

    dtrain = xgb.DMatrix(X)
    dtrain.set_float_info('label_lower_bound', np.tile(y_label['min.lambda'].values, ntile))
    dtrain.set_float_info('label_upper_bound', np.tile(y_label['max.lambda'].values, ntile))
    dtest = xgb.DMatrix(X_test)
    dtest.set_float_info('label_lower_bound', np.tile(y_label_test['min.lambda'].values, ntile))
    dtest.set_float_info('label_upper_bound', np.tile(y_label_test['max.lambda'].values, ntile))

    tstart = time.perf_counter()
    params[test_fold_id].update(base_params)
    bst = xgb.train(params[test_fold_id], dtrain, num_boost_round=nround[test_fold_id], evals=[(dtrain, 'train'), (dtest, 'test')])
    print(accuracy(bst.predict(dtest), dtest))
    print(f'Time elapsed = {time.perf_counter() - tstart} sec')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_method', required=True, choices=['hist', 'gpu_hist'])
    parser.add_argument('--test_fold_id', required=True, type=int)

    args = parser.parse_args()
    run(tree_method=args.tree_method, test_fold_id=args.test_fold_id)
