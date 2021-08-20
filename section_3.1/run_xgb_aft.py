import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold
import optuna
import time
import os
import json
import pathlib
import argparse
from multiprocessing import cpu_count
from data_importer import data_import

from optuna.samplers import RandomSampler, GridSampler, TPESampler

current_dir = os.path.dirname(os.path.abspath(__file__))

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
    return 'accuracy', acc

def train(trial, train_valid_folds, dtest, distribution, search_obj):
    params = search_obj.get_params(trial)
    params['aft_loss_distribution'] = distribution
    params.update(search_obj.get_base_params())

    bst = []  # bst[i]: XGBoost model fit using i-th CV fold
    best_iteration = 0
    best_score = float('-inf')
    max_round = 5000
    # Validation metric needs to improve at least once in every early_stopping_rounds rounds to
    # continue training.
    early_stopping_rounds = 25
    for dtrain, dvalid in train_valid_folds:
        bst.append(xgb.Booster(params, [dtrain, dvalid]))

    # Use CV metric to decide to early stop. CV metric = mean validation accuracy over CV folds
    for iteration_id in range(max_round):
        valid_metric = []
        for fold_id, (dtrain, dvalid) in enumerate(train_valid_folds):
            bst[fold_id].update(dtrain, iteration_id)
            msg = bst[fold_id].eval_set([(dvalid, 'valid')], iteration_id)
            valid_metric.append(float([x.split(':') for x in msg.split()][1][1]))
        cv_valid_metric = np.mean(valid_metric)
        if cv_valid_metric > best_score:
            best_score = cv_valid_metric
            best_iteration = iteration_id
        elif iteration_id - best_iteration >= early_stopping_rounds:
            # Early stopping
            break

    trial.set_user_attr('num_round', best_iteration)
    trial.set_user_attr('timestamp', time.perf_counter())

    return best_score


def run_nested_cv(inputs, labels, folds, seed, dataset_name, search_obj, n_trials, distributions, sampler, trial_log_fmt):
    fold_ids = np.unique(folds['fold'].values)

    for distribution in distributions:
        # Nested Cross-Validation, with 4-folds CV in the outer loop and 5-folds CV in the inner loop
        for test_fold_id in fold_ids:
            start = time.perf_counter()
            # train_valid_folds: list of form [(train_set, valid_set), ...], where train_set is used for training
            #                    and valid_set is used for model selection, i.e. hyperparameter search
            # dtest: held-out test set; will not be used for training or model selection
            kfold_gen = KFold(n_splits=5, shuffle=True, random_state=seed)
            train_valid_folds, dtrain_valid_combined, dtest \
              = get_train_valid_test_splits(folds, test_fold_id, inputs, labels, kfold_gen)

            study = optuna.create_study(sampler=sampler, direction='maximize')
            study.optimize(lambda trial : train(trial, train_valid_folds, dtest, distribution, search_obj),
                           n_trials=n_trials,
                           n_jobs=cpu_count() // 2)

            # Use the best hyperparameter set to fit a model with all data points except the held-out test set
            best_params = study.best_params
            best_num_round = study.best_trial.user_attrs['num_round']
            best_params.update(search_obj.get_base_params())
            best_params['aft_loss_distribution'] = distribution
            final_model = xgb.train(best_params, dtrain_valid_combined,
                                    num_boost_round=best_num_round,
                                    evals=[(dtrain_valid_combined, 'train_valid'), (dtest, 'test')],
                                    verbose_eval=False)

            # Evaluate accuracy on the test set
            # Accuracy = % of data points for which the final model produces a prediction that falls within the label range
            acc = accuracy(final_model.predict(dtest), dtest)
            time_taken = time.perf_counter() - start
            print(f'Fold {test_fold_id}, Distribution {distribution}: Accuracy {acc}, Time elapsed = {time_taken}')

            trial_log = trial_log_fmt.format(dataset_name=dataset_name, distribution=distribution, test_fold_id=test_fold_id)
            pathlib.Path(os.path.dirname(os.path.join(current_dir, trial_log))).mkdir(parents=True, exist_ok=True)

            trials = study.get_trials(deepcopy=False)
            timestamp = [trial.user_attrs['timestamp'] - start for trial in trials]

            with open(trial_log, 'w') as f:
                trials = study.get_trials(deepcopy=False)
                trial_id = [trial.number for trial in trials]
                score = [trial.value for trial in trials]
                timestamp = [trial.user_attrs['timestamp'] - start for trial in trials]
                json.dump({'trial_id': trial_id, 'cv_accuracy': score, 'timestamp': timestamp, 'final_accuracy': acc}, f)

        time_taken = time.perf_counter() - start
        print(f'Time elapsed = {time_taken}, distribution = {distribution}')



class HPO:
    def get_params(self, trial):
        eta              = trial.suggest_loguniform('learning_rate', 0.001, 1.0)
        max_depth        = trial.suggest_int('max_depth', 2, 10, step=2)
        min_child_weight = trial.suggest_loguniform('min_child_weight', 0.1, 100.0)
        reg_alpha        = trial.suggest_loguniform('reg_alpha', 0.0001, 100)
        reg_lambda       = trial.suggest_loguniform('reg_lambda', 0.0001, 100)
        sigma            = trial.suggest_loguniform('aft_loss_distribution_scale', 1.0, 100.0)
        return {'eta': eta,
                'max_depth': int(max_depth),
                'min_child_weight': min_child_weight,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'aft_loss_distribution_scale': sigma}

    def get_base_params(self):
        return {'verbosity': 0,
                'objective': 'survival:aft',
                'tree_method': 'hist',
                'nthread': 1,
                'eval_metric': 'interval-regression-accuracy'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', required=True, choices=['chipseq', 'simulated'])
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--n_trials', required=False, type=int, default=100)
    parser.add_argument('--seed', required=False, type=int, default=1)

    args = parser.parse_args()
    print(f'Dataset = {args.dataset}')
    print(f'Using {cpu_count() // 2} threads to run hyperparameter search')
    inputs, labels, folds = data_import(current_dir, args.data_source, args.dataset)

    run_nested_cv(inputs, labels, folds, seed=args.seed, dataset_name=args.dataset,
                  search_obj=HPO(), n_trials=args.n_trials, distributions=['normal', 'logistic', 'extreme'],
                  sampler=RandomSampler(seed=args.seed),
                  trial_log_fmt='xgb-aft-log/{dataset_name}/{distribution}-fold{test_fold_id}.json')

if __name__ == '__main__':
    main()

