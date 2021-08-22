"""
Want to determine the optimal split for regressing on the filtered dataset.
Also want to determine which regressor to use:

Regression Models:
    - GP (MLP or RQ Kernel)
    - ERT
    - ANN (Tabnet or FFNN)
"""


from data import utils0 as utils
import numpy as np
from tqdm import tqdm
# GPs
import GPy

# ERTs
from sklearn.ensemble import ExtraTreesRegressor

# ANN
import torch
from pytorch_tabnet.tab_model import TabNetRegressor # Tab Net

# model selection, CV k-fold
from sklearn.model_selection import RepeatedKFold
# Metrics
from sklearn.metrics import mean_squared_error
import argparse

# Plotting
import matplotlib.pyplot as plt

# Saving Data
import pickle

def cv_test_regressor(datasets, model_type='ERT', model_params={}):
    dataset, final_exam = datasets
    X, y = dataset
    X_exam, y_exam = final_exam
    if model_type in ['GP(MLP)','GP(RQ)', 'TabNet']:
        y_exam = np.expand_dims(y_exam, axis=1)

    CV = RepeatedKFold(n_splits=4, n_repeats=4)
    results = {'train': [], 'test': [], 'exam': []}

    for idx, (train, test) in enumerate(CV.split(X)):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        if model_type in ['GP(MLP)','GP(RQ)', 'TabNet']:
            y_train = np.expand_dims(y_train, axis=1)
            y_test = np.expand_dims(y_test, axis=1)

        reg = train_regressor(X_train, y_train, model_type, model_params)

        y_preds_train = reg.predict(X_train)
        y_preds_test = reg.predict(X_test)
        y_preds_exam = reg.predict(X_exam)
        if model_type in ['GP(MLP)','GP(RQ)']:
            y_preds_train = y_preds_train[0]
            y_preds_test = y_preds_test[0]
            y_preds_exam = y_preds_exam[0]

        train_RMSE = mean_squared_error(y_train, y_preds_train, squared=False)
        test_RMSE = mean_squared_error(y_test, y_preds_test, squared=False)
        exam_RMSE = mean_squared_error(y_exam, y_preds_exam, squared=False)

        results['train'].append(train_RMSE)
        results['test'].append(test_RMSE)
        results['exam'].append(exam_RMSE)


    scores = {'train': (np.mean(results['train']), np.std(results['train'])), 'test': (np.mean(results['test']), np.std(results['test'])), 'exam': (np.mean(results['exam']), np.std(results['exam']))}


    return scores


def train_regressor(X_train, y_train,  model_type = 'ERT', model_params={}):
    if model_type == 'ERT':
        reg = ExtraTreesRegressor(random_state=42, **model_params)
        reg.fit(X_train, y_train)

    elif model_type == 'GP(MLP)':
        kernel = GPy.kern.MLP(input_dim=10, ARD=True)
        reg = GPy.models.GPRegression(X_train, y_train, kernel)
        # reg.mlp.constrain_positive()
        reg.optimize(messages=False)

    elif model_type== 'GP(RQ)':

        # y_train = np.expand_dims(y_train, axis=1)
        # y_test = np.expand_dims(y_test, axis=1)

        kernel = GPy.kern.RQ(input_dim=10, ARD=True)
        reg = GPy.models.GPRegression(X_train, y_train, kernel)
        # reg.rq.constrain_positive()

        reg.optimize(messages=False)
    elif model_type == 'TabNet':
        reg = TabNetRegressor()
        reg.fit(X_train, y_train)
    else:
        raise(RuntimeError('Regressor Model type not allowed, please choose form GP(MLP), GP(RQ), ERT, or TabNet'))
    return reg


def main(**kwargs):
    print('\n------------ Split {} ------------'.format(kwargs['neped_split']))
    dataset, input_scaler = utils.load_data_torch(neped_split=kwargs['neped_split'], n_samples=5)
    low_neped, high_neped, final_exam = dataset

    scores_low = cv_test_regressor(datasets=(low_neped, final_exam), model_type=args.regression_model_type)
    scores_high = cv_test_regressor(datasets=(high_neped, final_exam), model_type=args.regression_model_type)

    results = {'LnM_train': scores_low['train'][0], 'HnM_train': scores_high['train'][0], 'LnM_train_std': scores_low['train'][1], 'HnM_train_std': scores_high['train'][1],
                'LnM_test': scores_low['test'][0], 'HnM_test': scores_high['test'][0], 'LnM_test_std': scores_low['test'][1], 'HnM_test_std': scores_high['test'][1],
                'LnM_exam': scores_low['exam'][0], 'HnM_exam': scores_high['exam'][0], 'LnM_exam_std': scores_low['exam'][1], 'HnM_exam_std': scores_high['exam'][1], }
    return results

import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def experiment(**kwargs):
    splits = np.linspace(3, 10, num=50)

    scores_final = {'HnM_test': [], 'LnM_test': [], 'HnM_train': [], 'LnM_train': [], 'HnM_exam': [], 'LnM_exam': [],
                    'HnM_test_std': [], 'LnM_test_std': [], 'HnM_train_std': [], 'LnM_train_std': [], 'HnM_exam_std': [], 'LnM_exam_std': []}
    iterator = tqdm(splits)
    for split in iterator:
        kwargs['neped_split'] = split
        with HiddenPrints():
            scores = main(**kwargs)
        iterator.set_postfix(scores)
        for key, val in scores.items():
            scores_final[key].append(val)

    file_name = './out/reg_split_' + kwargs['regression_model_type']  + '_cv_results.pickle'
    with open(file_name, 'wb') as file:
        pickle.dump(scores_final, file)


    fig1 = plt.figure()
    plt.plot(splits, scores_final['HnM_train'], label='High {} Model, train'.format(kwargs['regression_model_type']))
    plt.plot(splits, scores_final['LnM_train'], label='Low {} Model, train'.format(kwargs['regression_model_type']))
    plt.errorbar(splits, scores_final['HnM_train'], fmt='none', yerr=scores_final['HnM_train_std'], alpha=0.3, color='salmon')
    plt.errorbar(splits, scores_final['LnM_train'], fmt='none', yerr=scores_final['LnM_train_std'], alpha=0.3, color='black')
    plt.legend()

    fig2 = plt.figure()
    plt.plot(splits, scores_final['HnM_test'], label='High {} Model, test'.format(kwargs['regression_model_type']))
    plt.plot(splits, scores_final['LnM_test'], label='Low {} Model, test'.format(kwargs['regression_model_type']))
    plt.errorbar(splits, scores_final['HnM_test'], fmt='none', yerr=scores_final['HnM_test_std'], alpha=0.3, color='salmon')
    plt.errorbar(splits, scores_final['LnM_test'], fmt='none', yerr=scores_final['LnM_test_std'], alpha=0.3, color='black')
    plt.legend()

    fig3 = plt.figure()
    plt.plot(splits, scores_final['HnM_exam'], label='High {} Model, exam'.format(kwargs['regression_model_type']))
    plt.plot(splits, scores_final['LnM_exam'], label='Low {} Model, exam'.format(kwargs['regression_model_type']))
    plt.errorbar(splits, scores_final['HnM_exam'], fmt='none', yerr=scores_final['HnM_exam_std'], alpha=0.3, color='salmon')
    plt.errorbar(splits, scores_final['LnM_exam'], fmt='none', yerr=scores_final['LnM_exam_std'], alpha=0.3, color='black')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split Regress Experiment, finding optimal split and regressor')
    parser.add_argument("-rmt", "--regression_model_type", help='Which regression model to use', default='ERT', type=str)
    parser.add_argument("-split", "--neped_split", help='Where to split', default=9.5, type=float)

    args = parser.parse_args()
    config = vars(args)
    experiment(**config)

    # main(**config)
