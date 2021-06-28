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
import scipy as sp

import pickle

import matplotlib.pyplot as plt
import seaborn as sns
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
        coef_left = reg1.feature_importances_
        coef_right = reg2.feature_importances_
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
        coef_left =  lin1.coef_
        coef_right = lin2.coef_
    # print(np.abs(lin1.coef_ - lin2.coef_))
    return coef_left, coef_right

def main(inputs, targets, dict_splits, reggresor='linear'):
    coefs_left, coefs_right = [], []
    for key, item in dict_splits.items():
        print('\n # {}: {}'.format(key, item))
        datasplits = utils.create_joint_distr(inputs, targets, col=key, split=item)
        new_left, new_right = split_and_fit(datasplits, reggresor)
        coefs_left.append(new_left)
        coefs_right.append(new_right)

    return coefs_left, coefs_right
    # plot_corr(matrix)

def get_kde(df, col_string, show_results=True):
    series = df[col_string]
    xmin, xmax = min(series), max(series)
    points = np.linspace(xmin, xmax, num=1000)
    kernel = sp.stats.gaussian_kde(series)
    Y = kernel(points)
    ymax_i = np.argmax(Y)
    ymax = Y[ymax_i]
    if show_results:
        fig = plt.figure(figsize=(18, 18))
        plt.plot(points, Y, lw=5)
        plt.vlines(points[ymax_i], min(Y), ymax, color='black', ls='--', lw=5)
        plt.annotate('Split at {:.3}'.format(points[ymax_i]), xy=(0.7, 0.8), xycoords='figure fraction')
        plt.title(col_string + ' KDE')
        plt.show()

    return points[ymax_i]

def get_split_from_kde(df, col_list, show_results=False):
    dict_splits = {}
    for col in col_list:
        split_point = get_kde(df, col, show_results)
        dict_splits[col] = split_point
    return dict_splits

def plot_coefs(col, split, i):
    fig = plt.figure(figsize=(18, 18))
    plt.bar(np.arange(0, len(all_coefs[0])) - 0.25, all_coefs[0], width=0.25, label='No split')
    plt.bar(np.arange(0, len(all_coefs[0])) + 0.25, left_split[i][0], width=0.25,label='> {:.4}'.format(split))
    plt.bar(np.arange(0, len(all_coefs[0])), right_split[i][0], width=0.25, label='$\leq$ {:.4}'.format(split))
    plt.xticks(np.arange(0, len(all_coefs[0])), main_eng_latex)
    plt.title('{}'.format(col))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    df_sep = pd.read_csv('/home/adam/data/seperatrix_dataset.csv')
    main_eng_latex = ['$I_P$', '$B_T$','$a$', '$\delta$',
    '$P_{NBI}$', '$P_{ICRH}$', '$V_P$', '$q_{95}$', '$\Gamma$']
    splits = [2.5, 2.5, 0.915, 0.322, 15, 1.1, 77.5, 3.657, 2.5]
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
    input_scaler = StandardScaler()
    input_scale = input_scaler.fit_transform(input_df)
    all_reg = LinearRegression()
    all_reg.fit(input_scale, target_df)
    all_coefs = all_reg.coef_
    """
    Get split values for each
    """
    dict_splits = get_split_from_kde(input_df, main_eng_latex, show_results=False)
    print(dict_splits)
    # matrix = main(input_df, target_df, dict_splits, reggresor='forest')
    left_coefs, right_coefs = main(input_df, target_df, dict_splits, reggresor='linear')
    result_dict = {'dict_split': dict_splits, 'all_coefs': all_coefs, 'left_split_c': left_coefs, 'right_split_c': right_coefs}
    print(result_dict)
    with open('/home/adam/ENR_Sven/greedy-pillows/src/out/splits/linear-coef-28-06-2021.pickle', 'wb') as file:
        pickle.dump(result_dict, file)

    dict_split = result_dict['dict_split']
    all_coefs = result_dict['all_coefs']
    left_split = result_dict['left_split_c']
    right_split = result_dict['right_split_c']
    i = 0
    for col, split in dict_split.items():
        print(col, split)
        series = df_sep[col]
        xmin, xmax = min(series), max(series)
        points = np.linspace(xmin, xmax, num=1000)
        kernel = sp.stats.gaussian_kde(series)
        Y = kernel(points)
        ymax_i = np.argmax(Y)
        ymax = Y[ymax_i]
        fig, axs = plt.subplots(1, 2, figsize=(36, 18))
        axs[0].plot(points, Y, lw=5)
        axs[0].fill_between(points[ymax_i:], Y[ymax_i:], interpolate=True, color='salmon')
        axs[0].fill_between(points[:ymax_i], Y[:ymax_i], interpolate=True, color='green')
        axs[0].vlines(points[ymax_i], min(Y), ymax, color='black', ls='--', lw=5)
        axs[0].annotate('Split at {:.3}'.format(points[ymax_i]), xy=(0.7, 0.8), xycoords='figure fraction')
        axs[0].set(xlabel=col, ylabel='KDE')

        axs[1].bar(np.arange(0, len(all_coefs[0])) - 0.25, all_coefs[0], width=0.25, label='No split')
        axs[1].bar(np.arange(0, len(all_coefs[0])) + 0.25, left_split[i][0], width=0.25,label='> {:.4}'.format(split), color='salmon')
        axs[1].bar(np.arange(0, len(all_coefs[0])), right_split[i][0], width=0.25, label='$\leq$ {:.4}'.format(split), color='green')
        axs[1].set(xticks=(np.arange(0, len(all_coefs[0]))), xticklabels=main_eng_latex, ylabel='Linear Coefficients')
        axs[1].legend()
        plt.suptitle('$n_e^{{sep}}$ Lin. Coef. Dependence on Data Split via {}'.format(col))
        plt.savefig('/home/adam/ENR_Sven/greedy-pillows/src/out/splits/KDE_vs_COEF-{}'.format(col))
        plt.show()
        i += 1


    # matricies = [matrix, matrix2]


    # plot_corr(matricies, main_eng_latex)




    """
    all_df = df_sep[seperatrix_profiles_latex + main_eng_latex]
    g = sns.PairGrid(all_df, diag_sharey=False, corner=True)
    g.map_lower(sns.scatterplot)
    g.map_diag(sns.kdeplot)
    plt.show()
    """
    """
    dict_splits = {key: value for key ,value in zip(main_eng_latex, splits)}

    matrix = main(input_df, target_df, dict_splits, reggresor='forest')
    matrix2 = main(input_df, target_df, dict_splits, reggresor='linear')

    matricies = [matrix, matrix2]


    plot_corr(matricies, main_eng_latex)
    """
    # plot_corr(matrix, main_eng_latex)

    # print(matrix)
