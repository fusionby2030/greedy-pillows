"""
The experiment is to split the dataset different subsets based
on the joint distributions of input variables and check how prediction changes

First we examine how the coefficints of a linear regressor change
"""

from codebase import utils
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import pickle
SMALL_SIZE = 40
MEDIUM_SIZE = 45
BIGGER_SIZE = 50

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plot_corr(matrix, labels):
    if len(matrix) == 2:
        fig, axs = plt.subplots(1, 2, figsize=(36, 18))
        data1, data2 = matrix[0], matrix[1]
        im1 = axs[0].imshow(data1, cmap='viridis')
        im2 = axs[1].imshow(data2, cmap='viridis')

        # add space for colour bar
        fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
        fig.colorbar(im1, ax=axs[0])
        fig.colorbar(im2, ax=axs[1])
        axs[0].set(title='Random Forest')
        axs[1].set(title='Linear Regressor')
        plt.setp(axs, xticks=[i for i in range(0, len(labels))], xticklabels=labels, yticks=[i for i in range(0, len(labels))], yticklabels=labels, ylabel='Split Variable', xlabel='Diff. of Importances')

    else:
        fig, axs = plt.subplots(1, 1, figsize=(36, 18))
        im = axs.imshow(matrix)
        axs.set(xticks=[i for i in range(0, len(labels))], xticklabels=labels, yticks=[i for i in range(0, len(labels))], yticklabels=labels)
        fig.colorbar(im)
    plt.tight_layout()
    plt.show()

def split_and_fit(datasplits, regressor):
    X1, y1, X2, y2 = datasplits
    if regressor=='forest':
        y1 = y1.to_numpy().ravel()
        y2 = y2.to_numpy().ravel()
        reg1 = RandomForestRegressor()
        reg2 = RandomForestRegressor()
        reg1.fit(X1, y1)
        reg2.fit(X2, y2)

        print(mean_squared_error(squared=False, y_true=y2, y_pred=reg1.predict(X2)))
        print(mean_squared_error(squared=False, y_true=y1, y_pred=reg2.predict(X1)))
        diff = np.abs(reg1.feature_importances_, reg2.feature_importances_)
    else:
        X_scale = StandardScaler()
        X1 = X_scale.fit_transform(X1)
        X2 = X_scale.transform(X2)

        lin1 = LinearRegression()
        lin2= LinearRegression()

        lin1.fit(X1, y1)
        lin2.fit(X2, y2)
        print(mean_squared_error(squared=False, y_true=y2, y_pred=lin1.predict(X2)))
        print(mean_squared_error(squared=False, y_true=y1, y_pred=lin2.predict(X1)))
        diff = np.abs(lin1.coef_ - lin2.coef_)
    # print(np.abs(lin1.coef_ - lin2.coef_))
    return diff

def main(inputs, targets, dict_splits, reggresor='linear'):
    matrix = []
    for key, item in dict_splits.items():
        print('\n # {}: {}'.format(key, item))
        datasplits = utils.create_joint_distr(inputs, targets, col=key, split=item)
        new_coefs = split_and_fit(datasplits, reggresor)
        matrix.append(new_coefs.ravel())

    return matrix
    # plot_corr(matrix)

import seaborn as sns

if __name__ == '__main__':
    df_sep = pd.read_csv('/home/adam/data/seperatrix_dataset.csv')
    main_eng_latex = ['$I_P$', '$B_T$','$a$', '$\delta$',
    '$P_{NBI}$', '$P_{ICRH}$', '$V_P$', '$q_{95}$', '$\Gamma$']
    main_engineer = ['Ip(MA)', 'B(T)', 'a(m)', 'averagetriangularity',
                 'P_NBI(MW)', 'P_ICRH(MW)',
                 'plasmavolume(m3)', 'q95', 'gasflowrateofmainspecies1022(es)']

    seperatrix_profiles = ['neseparatrixfromexpdata1019(m-3)']
    seperatrix_profiles_latex = ['$n_e^{sep}$']

    rename_dict = {key: value for key, value in zip(main_engineer + seperatrix_profiles, main_eng_latex + seperatrix_profiles_latex)}
    # rename_dict[seperatrix_profiles[0]] = seperatrix_profiles_latex[0]
    df_sep.rename(columns=rename_dict, inplace=True)


    input_df = df_sep[main_eng_latex]
    target_df = df_sep[seperatrix_profiles_latex]
    """all_df = df_sep[seperatrix_profiles_latex + main_eng_latex]
    g = sns.PairGrid(all_df, diag_sharey=False, corner=True)
    g.map_lower(sns.scatterplot)
    g.map_diag(sns.kdeplot)
    plt.show()

    """
    splits = [2.5, 2.5, 0.9, 0.322, 15, 1.1, 77.5, 3.657, 2.5]
    dict_splits = {key: value for key ,value in zip(main_eng_latex, splits)}

    matrix = main(input_df, target_df, dict_splits, reggresor='forest')
    matrix2 = main(input_df, target_df, dict_splits, reggresor='linear')

    matricies = [matrix, matrix2]


    plot_corr(matricies, main_eng_latex)
    # plot_corr(matrix, main_eng_latex)

    # print(matrix)
