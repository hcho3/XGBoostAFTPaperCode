import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import optuna
import os
import boto3
import time
import json
import argparse
from multiprocessing import cpu_count

from optuna.samplers import RandomSampler, GridSampler, TPESampler

def data_import(data_name):
    filename = 'https://raw.githubusercontent.com/avinashbarnwal/GSOC-2019/master/AFT/test/data/'+data_name+'/'
    inputFileName = filename + 'inputs.csv'
    labelFileName = filename + 'outputs.csv'
    foldsFileName = filename + 'cv/equal_labels/folds.csv'
    inputs        = pd.read_csv(inputFileName, index_col='sequenceID')
    labels        = pd.read_csv(labelFileName, index_col='sequenceID')
    folds         = pd.read_csv(foldsFileName, index_col='sequenceID')
    res           = {}
    res['inputs'] = inputs
    res['labels'] = labels
    res['folds']  = folds
    return(res)

def preprocess_data(inputs,labels):
    inputs.replace([-float('inf'), float('inf')], np.nan, inplace=True)
    missingCols = inputs.isnull().sum()
    missingCols = list(missingCols[missingCols > 0].index)
    print(f'missingCols = {missingCols}')

    inputs.drop(missingCols,axis=1,inplace=True)

    varCols     = inputs.apply(lambda x: np.var(x))
    zeroVarCols = list(varCols[varCols==0].index)
    print(f'zeroVarCols = {zeroVarCols}')

    inputs.drop(zeroVarCols,axis=1,inplace=True)
    labels['min.lambda'] = labels['min.log.lambda'].apply(lambda x: np.exp(x))
    labels['max.lambda'] = labels['max.log.lambda'].apply(lambda x: np.exp(x))

    return inputs, labels

def get_train_valid_test_splits(folds, test_fold_id, inputs, labels, kfold_gen):
    # Split data into train and test
    X            = inputs[folds['fold'] != test_fold_id].values
    X_test       = inputs[folds['fold'] == test_fold_id].values
    y_label      = labels[folds['fold'] != test_fold_id]
    y_label_test = labels[folds['fold'] == test_fold_id]

    dtest = xgb.DMatrix(X_test)
    dtest.set_float_info('label_lower_bound', y_label_test['min.lambda'].values)
    dtest.set_float_info('label_upper_bound', y_label_test['max.lambda'].values)

    # Further split train into train and valid. Do this 5 times to obtain 5 fold cross validation
    folds = []
    dmat_train_valid_combined = xgb.DMatrix(X)
    dmat_train_valid_combined.set_float_info('label_lower_bound', y_label['min.lambda'].values)
    dmat_train_valid_combined.set_float_info('label_upper_bound', y_label['max.lambda'].values)
    for train_idx, valid_idx in kfold_gen.split(X):
        dtrain = xgb.DMatrix(X[train_idx, :])
        dtrain.set_float_info('label_lower_bound', y_label['min.lambda'].values[train_idx])
        dtrain.set_float_info('label_upper_bound', y_label['max.lambda'].values[train_idx])

        dvalid = xgb.DMatrix(X[valid_idx, :])
        dvalid.set_float_info('label_lower_bound', y_label['min.lambda'].values[valid_idx])
        dvalid.set_float_info('label_upper_bound', y_label['max.lambda'].values[valid_idx])

        folds.append((dtrain, dvalid))

    return (folds, dmat_train_valid_combined, dtest)


def accuracy(predt, dmat):
    y_lower = dmat.get_float_info('label_lower_bound')
    y_upper = dmat.get_float_info('label_upper_bound')
    acc = np.sum((predt >= y_lower) & (predt <= y_upper)) / len(predt)
    return acc

base_params =  {'verbosity': 0,
                'objective': 'survival:aft',
                'tree_method': 'hist',
                'eval_metric': 'interval-regression-accuracy'}

def train(trial, train_valid_folds, dtrain_valid_combined, dtest, distribution):
    learning_rate    = trial.suggest_loguniform('learning_rate', 0.001, 1.0)
    max_depth        = trial.suggest_int('max_depth', 2, 10, step=1)
    min_child_weight = trial.suggest_loguniform('min_child_weight', 0.001, 100.0)
    reg_alpha        = trial.suggest_loguniform('reg_alpha', 0.001, 100.0)
    reg_lambda       = trial.suggest_loguniform('reg_lambda', 0.001, 100.0)
    sigma            = trial.suggest_uniform('aft_loss_distribution_scale', 0.5, 2.0)
    params = {'learning_rate': learning_rate,
              'max_depth': int(max_depth),
              'min_child_weight': min_child_weight,
              'reg_alpha': reg_alpha,
              'reg_lambda': reg_lambda,
              'aft_loss_distribution_scale': sigma}
    params['aft_loss_distribution'] = distribution
    params.update(base_params)

    # Cross validation metric is computed as follows:
    # 1. For each of the 5 folds, run XGBoost for 5000 rounds and record trace of the validation metric (accuracy)
    # 2. Compute the mean validation metric over the 5 folds, for each iteration ID.
    # 3. Select the iteration ID which maximizes the mean validation metric.
    # 4. Return the mean validation metric as CV metric.
    validation_metric_history = pd.DataFrame()
    for fold_id, (dtrain, dvalid) in enumerate(train_valid_folds):
        res = {}
        bst = xgb.train(params, dtrain, num_boost_round=5000,
                        evals=[(dtrain, 'train'), (dvalid, 'valid')],
                        verbose_eval=False, evals_result=res)
        validation_metric_history[fold_id] = res['valid']['interval-regression-accuracy']
    validation_metric_history['mean'] = validation_metric_history.mean(axis=1)
    best_num_round = validation_metric_history['mean'].idxmax()

    trial.set_user_attr('num_round', best_num_round)
    trial.set_user_attr('timestamp', time.time())

    # Use the hyperparameter set to fit new model with all data points except the held-out test set
    # Then compute test accuracy. Note: test accuracy is NOT be used for model selection!
    final_model = xgb.train(params, dtrain_valid_combined, num_boost_round=best_num_round,
                            evals=[(dtrain_valid_combined, 'train-valid'), (dtest, 'test')],
                            verbose_eval=False)
    test_acc = accuracy(final_model.predict(dtest), dtest)
    trial.set_user_attr('test_acc', test_acc)

    # Use validation accuracy to judge hyperparameter set
    return validation_metric_history.iloc[best_num_round].mean()


def run_hpo_nested_cv(inputs, labels, folds, args):
    if args.sampler == 'random':
        sampler = RandomSampler(seed=args.seed)
    elif args.sampler == 'tpe':
        sampler = TPESampler(seed=args.seed)
    else:
        raise ValueError(f'Unknown sampler: {args.sampler}')

    # Nested Cross-Validation, with 4-folds CV in the outer loop and 5-folds CV in the inner loop
    # Outer test fold is set by script argument
    start = time.time()
    # train_valid_folds: list of form [(train_set, valid_set), ...], where train_set is used for training
    #                    and valid_set is used for model selection, i.e. hyperparameter search
    # dtest: held-out test set; will not be used for training or model selection
    kfold_gen = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_valid_folds, dtrain_valid_combined, dtest \
      = get_train_valid_test_splits(folds, args.test_fold_id, inputs, labels, kfold_gen)

    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(lambda trial : train(trial, train_valid_folds, dtrain_valid_combined, dtest,
                                        args.distribution),
                   n_trials=args.ntrial, n_jobs=args.nthread)

    best_params = study.best_params
    best_num_round = study.best_trial.user_attrs['num_round']
    best_params.update(base_params)
    best_params['aft_loss_distribution'] = args.distribution
    final_model = xgb.train(best_params, dtrain_valid_combined, num_boost_round=best_num_round,
                            evals=[(dtrain_valid_combined, 'train-valid'), (dtest, 'test')],
                            verbose_eval=False)

    # Evaluate accuracy on the test set
    # Accuracy = % of data points for which the final model produces a prediction that falls within the label range
    acc = accuracy(final_model.predict(dtest), dtest)
    print(f'Fold {args.test_fold_id}: Accuracy {acc}')
    model_file = f'{args.sampler}-{args.dataset}-{args.test_fold_id}-{args.distribution}-model.json'
    final_model.save_model(model_file)

    trial_log = f'{args.sampler}-{args.dataset}-{args.test_fold_id}-{args.distribution}.json'
    with open(trial_log, 'w') as f:
        trials = study.get_trials(deepcopy=False)
        trial_id = [trial.number for trial in trials]
        score = [trial.value for trial in trials]
        timestamp = [trial.user_attrs['timestamp'] - start for trial in trials]
        test_acc = [trial.user_attrs['test_acc'] for trial in trials]
        json.dump({'trial_id': trial_id, 'valid_acc': score, 'test_acc': test_acc,
                   'timestamp': timestamp, 'final_accuracy': acc}, f)

    print(f'Uploading {model_file} and {trial_log} to S3 bucket {args.s3_bucket}...')
    boto3.resource('s3').Bucket(args.s3_bucket).upload_file(model_file, model_file)
    boto3.resource('s3').Bucket(args.s3_bucket).upload_file(trial_log, trial_log)

    end = time.time()
    time_taken = end - start
    print(f'Time elapsed = {time_taken}, distribution = {args.distribution}, ' +
          f'test_fold_id = {args.test_fold_id}')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--distribution', required=True, choices=['normal', 'logistic', 'extreme'])
    parser.add_argument('--test_fold_id', required=True, type=int, choices=[1, 2, 3, 4])
    parser.add_argument('--seed', required=False, type=int, default=1)
    parser.add_argument('--sampler', required=True, choices=['random', 'tpe'])
    parser.add_argument('--nthread', required=True, type=int)
    parser.add_argument('--ntrial', required=True, type=int)
    parser.add_argument('--s3_bucket', required=True)

    args = parser.parse_args()
    print(f'Dataset = {args.dataset}')
    print(f'Using {args.nthread} threads to run hyperparameter search')
    data  = data_import(args.dataset)
    inputs = data['inputs']
    labels = data['labels']
    folds  = data['folds']

    print(f'sampler = {args.sampler}, distribution = {args.distribution}, ' +
          f'test_fold_id = {args.test_fold_id}, # trials = {args.ntrial}')

    inputs, labels = preprocess_data(inputs, labels)

    run_hpo_nested_cv(inputs, labels, folds, args)

if __name__ == '__main__':
    main()
