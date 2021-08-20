import numpy as np
import scipy.optimize as opt
import xgboost as xgb
import time
import argparse
import sys
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import KFold
import optuna
from optuna.samplers import RandomSampler


def xgb_train(*, trial, train_valid_folds, model_obj, valid_metric_func):
    params = model_obj.get_params(trial)
    params.update(model_obj.get_base_params())

    bst = []  # bst[i]: XGBoost model fit using i-th CV fold
    best_iteration = 0
    best_score = float('-inf')
    max_round = 500
    # Validation metric needs to improve at least once in every early_stopping_rounds rounds to
    # continue training.
    early_stopping_rounds = 5
    for dtrain, dvalid, _ in train_valid_folds:
        bst.append(xgb.Booster(params, [dtrain, dvalid]))

    # Use CV metric to decide to early stop. CV metric = mean validation accuracy over CV folds
    for iteration_id in range(max_round):
        valid_metric = []
        for fold_id, (dtrain, dvalid, y_valid) in enumerate(train_valid_folds):
            bst[fold_id].update(dtrain, iteration_id)
            y_pred = bst[fold_id].predict(dvalid)
            valid_metric.append(valid_metric_func(y_valid, y_pred))
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


def xgb_compute_test_pred(*, study, dtrain_valid_combined, dtest):
    best_params = study.best_params
    best_num_round = study.best_trial.user_attrs['num_round']
    best_params.update(model_obj.get_base_params())
    final_model = xgb.train(best_params, dtrain_valid_combined,
                            num_boost_round=best_num_round,
                            evals=[(dtrain_valid_combined, 'train_valid'), (dtest, 'test')],
                            verbose_eval=False)
    y_pred = final_model.predict(dtest)
    return y_pred


class XGBoostAFT:
    def __init__(self, *, distribution):
        self.distribution = distribution

    def get_params(self, trial):
        eta = trial.suggest_loguniform('learning_rate', 0.001, 1.0)
        max_depth = trial.suggest_int('max_depth', 2, 10, step=2)
        min_child_weight = trial.suggest_loguniform('min_child_weight', 0.1, 100.0)
        reg_alpha = trial.suggest_loguniform('reg_alpha', 0.0001, 100)
        reg_lambda = trial.suggest_loguniform('reg_lambda', 0.0001, 100)
        sigma = trial.suggest_loguniform('aft_loss_distribution_scale', 1.0, 100.0)
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
                'aft_loss_distribution': self.distribution,
                'eval_metric': 'aft-nloglik'}

    def dmat_builder(self, X, y):
        label_lower_bound = np.array([e[1] for e in y])
        label_upper_bound = np.array([(e[1] if e[0] else +np.inf) for e in y])
        return xgb.DMatrix(X, label_lower_bound=label_lower_bound,
                           label_upper_bound=label_upper_bound)

    def estimated_risk(self, y_pred):
        return -y_pred

    def train(self, *, trial, train_valid_folds, model_obj, valid_metric_func):
        return xgb_train(trial=trial, train_valid_folds=train_valid_folds,
                         model_obj=model_obj, valid_metric_func=valid_metric_func)

    def compute_test_pred(self, *, study, dtrain_valid_combined, dtest):
        return xgb_compute_test_pred(study=study, dtrain_valid_combined=dtrain_valid_combined,
                                     dtest=dtest)


class XGBoostCox:
    def get_params(self, trial):
        eta = trial.suggest_loguniform('learning_rate', 0.001, 1.0)
        max_depth = trial.suggest_int('max_depth', 2, 10, step=2)
        min_child_weight = trial.suggest_loguniform('min_child_weight', 0.1, 100.0)
        reg_alpha = trial.suggest_loguniform('reg_alpha', 0.0001, 100)
        reg_lambda = trial.suggest_loguniform('reg_lambda', 0.0001, 100)
        return {'eta': eta,
                'max_depth': int(max_depth),
                'min_child_weight': min_child_weight,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda}

    def get_base_params(self):
        return {'verbosity': 0,
                'objective': 'survival:cox',
                'tree_method': 'hist',
                'eval_metric': 'cox-nloglik'}

    def dmat_builder(self, X, y):
        label = np.array([(e[1] if e[0] else -e[1]) for e in y])
        return xgb.DMatrix(X, label=label)

    def estimated_risk(self, y_pred):
        return y_pred

    def train(self, trial, train_valid_folds, model_obj, valid_metric_func):
        return xgb_train(trial=trial, train_valid_folds=train_valid_folds,
                         model_obj=model_obj, valid_metric_func=valid_metric_func)

    def compute_test_pred(self, *, study, dtrain_valid_combined, dtest):
        return xgb_compute_test_pred(study=study, dtrain_valid_combined=dtrain_valid_combined,
                                     dtest=dtest)


class SkSurvCoxLinear:
    def get_params(self, trial):
        alpha = trial.suggest_loguniform('alpha', 0.0001, 100)
        return {'alpha': alpha}

    def get_base_params(self):
        pass

    def dmat_builder(self, X, y):
        return X, y

    def estimated_risk(self, y_pred):
        return y_pred

    def train(self, trial, train_valid_folds, model_obj, valid_metric_func):
        params = model_obj.get_params(trial)

        valid_metric = []
        for dtrain, dvalid, _ in train_valid_folds:
            clf = CoxPHSurvivalAnalysis(alpha=params['alpha'], ties='breslow', n_iter=100, tol=1e-9)
            X_train, y_train = dtrain
            X_valid, y_valid = dvalid
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_valid)
            valid_metric.append(valid_metric_func(y_valid, y_pred))
        cv_valid_metric = np.mean(valid_metric)

        trial.set_user_attr('timestamp', time.perf_counter())

        return cv_valid_metric

    def compute_test_pred(self, *, study, dtrain_valid_combined, dtest):
        best_params = study.best_params
        final_model = CoxPHSurvivalAnalysis(alpha=best_params['alpha'], ties='breslow', n_iter=100,
                                            tol=1e-9)
        X_train_valid, y_train_valid = dtrain_valid_combined
        X_test, y_test = dtest
        final_model.fit(X_train_valid, y_train_valid)
        y_pred = final_model.predict(X_test)
        return y_pred


def f(x):
    return x[:, 0] + 3 * x[:, 2]**2 - 2 * np.exp(-x[:, 4])


def generate_survival_data(*, data_gen, n_samples, n_features, hazard_ratio, baseline_hazard,
                           censoring_fraction, rng):
    X = rng.uniform(low=0.0, high=1.0, size=(n_samples, n_features))
    risk = f(X) + rng.normal(loc=0.0, scale=0.3, size=(n_samples,))
    if data_gen == 'coxph':
        u = rng.uniform(low=0.0, high=1.0, size=n_samples)
        time_event = -np.log(u) / (baseline_hazard * np.power(hazard_ratio, risk))
    elif data_gen == 'aft':
        time_event = np.exp(-risk)
    else:
        raise Exception(f'Unknown data generator {args.data_gen}')

    def get_observed_time(upper_limit_censored_time):
        rng_cens = np.random.Generator(np.random.PCG64(seed=0))
        time_censor = rng_cens.uniform(low=0.0, high=upper_limit_censored_time, size=n_samples)
        event = time_event < time_censor
        time = np.where(event, time_event, time_censor)
        return event, time

    def censoring_objective(upper_limit_censored_time):
        event, _ = get_observed_time(upper_limit_censored_time)
        actual_censoring_fraction = 1.0 - event.sum() / event.shape[0]
        return (actual_censoring_fraction - censoring_fraction)**2

    res = opt.minimize_scalar(censoring_objective, method='bounded',
                              bounds=(0, time_event.max()))
    event, time = get_observed_time(res.x)
    tau = np.percentile(time[event], q=80)
    y = Surv.from_arrays(event=event, time=time)

    return X, y, tau


def get_train_valid_test_splits(*, X_train_valid, y_train_valid, X_test, y_test, inner_kfold_gen,
                                dmat_builder):
    dtest = dmat_builder(X_test, y_test)

    # Split remaining data into train and validation sets.
    # Do this 5 times to obtain 5-fold cross validation
    train_valid_ls = []
    dmat_train_valid_combined = dmat_builder(X_train_valid, y_train_valid)
    for train_idx, valid_idx in inner_kfold_gen.split(X_train_valid):
        dtrain = dmat_builder(X_train_valid[train_idx, :], y_train_valid[train_idx])
        dvalid = dmat_builder(X_train_valid[valid_idx, :], y_train_valid[valid_idx])
        train_valid_ls.append((dtrain, dvalid, y_train_valid[valid_idx]))

    return train_valid_ls, dmat_train_valid_combined, dtest


def run_nested_cv(*, X, y, tau, seed, sampler, n_trials, model_obj):
    def valid_metric_func(y_valid, y_pred):
        try:
            return concordance_index_ipcw(survival_train=y, survival_test=y_valid,
                                          estimate=model_obj.estimated_risk(y_pred), tau=tau)[0]
        except ValueError as e:
            return float('-inf')  # y_pred contains NaN or Inf, ensure that this model gets ignored
    # Nested Cross-Validation with 4-folds CV in the outer loop and 5-folds CV in the inner loop
    start = time.time()
    outer_kfold_gen = KFold(n_splits=4, shuffle=True, random_state=seed)
    for test_fold_id, (train_valid_idx, test_idx) in enumerate(outer_kfold_gen.split(X, y)):
        # train_valid_folds: list of form [(train_set, valid_set), ...], where train_set is used
        #                    for training and valid_set is used for model selection,
        #                    i.e. hyperparameter search
        # dtest: held-out test set; will not be used for training or model selection
        inner_kfold_gen = KFold(n_splits=5, shuffle=True, random_state=seed)
        train_valid_folds, dtrain_valid_combined, dtest \
            = get_train_valid_test_splits(X_train_valid=X[train_valid_idx, :],
                                          y_train_valid=y[train_valid_idx],
                                          X_test=X[test_idx, :],
                                          y_test=y[test_idx],
                                          inner_kfold_gen=inner_kfold_gen,
                                          dmat_builder=model_obj.dmat_builder)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(
            lambda trial: model_obj.train(trial=trial, train_valid_folds=train_valid_folds,
                                          model_obj=model_obj, valid_metric_func=valid_metric_func),
            n_trials=n_trials)

        # Use the best hyperparameter set to fit a model with all data points except the
        # held-out test set
        # Then evaluate C-index on the test set
        y_pred = model_obj.compute_test_pred(
            study=study, dtrain_valid_combined=dtrain_valid_combined, dtest=dtest)
        c_uno = concordance_index_ipcw(survival_train=y, survival_test=y[test_idx],
                                       estimate=model_obj.estimated_risk(y_pred), tau=tau)[0]
        print(f'Fold {test_fold_id}: C-statistics {c_uno}')

    end = time.time()
    time_taken = end - start
    print(f'Time elapsed = {time_taken}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=False, type=int, default=1)
    parser.add_argument('--n_samples', required=False, type=int, default=1000)
    parser.add_argument('--n_features', required=False, type=int, default=20)
    parser.add_argument('--n_trials', required=True, type=int)
    parser.add_argument('--censoring_fraction', required=True, type=float)
    parser.add_argument('--data_gen', required=True, type=str, choices=['coxph', 'aft'])
    parser.add_argument('--method', required=True, type=str,
                        choices=['XGBoostAFTNormal', 'XGBoostAFTLogistic', 'XGBoostAFTExtreme',
                                 'XGBoostCox', 'SkSurvCoxLinear'])
    args = parser.parse_args()

    rng = np.random.Generator(np.random.PCG64(seed=args.seed))
    X, y, tau = generate_survival_data(data_gen=args.data_gen, n_samples=1000, n_features=20,
                                       hazard_ratio=2, baseline_hazard=0.1,
                                       censoring_fraction=args.censoring_fraction, rng=rng)

    if args.method == 'XGBoostAFTNormal':
        model_obj = XGBoostAFT(distribution='normal')
    elif args.method == 'XGBoostAFTLogistic':
        model_obj = XGBoostAFT(distribution='logistic')
    elif args.method == 'XGBoostAFTExtreme':
        model_obj = XGBoostAFT(distribution='extreme')
    elif args.method == 'XGBoostCox':
        model_obj = XGBoostCox()
    elif args.method == 'SkSurvCoxLinear':
        model_obj = SkSurvCoxLinear()
    else:
        raise Exception(f'Unknown method {args.method}')

    run_nested_cv(X=X, y=y, tau=tau, seed=args.seed,
                  sampler=RandomSampler(seed=args.seed),
                  n_trials=args.n_trials,
                  model_obj=model_obj)
