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
def main(datasplits):
    X1, y1, X2, y2 = datasplits
    X_scale = StandardScaler()
    X1 = X_scale.fit_transform(X1)
    X2 = X_scale.transform(X2)

    lin1 = LinearRegression()
    lin2= LinearRegression()

    lin1.fit(X1, y1)
    lin2.fit(X2, y2)

    # print(np.abs(lin1.coef_ - lin2.coef_))
    return np.abs(lin1.coef_ - lin2.coef_)

import matplotlib.pyplot as plt
if __name__ == '__main__':
    df_sep = pd.read_csv('/home/adam/data/seperatrix_dataset.csv')
    main_engineer = ['Ip(MA)', 'B(T)', 'a(m)', 'averagetriangularity',
                'Meff', 'P_NBI(MW)',
                 'plasmavolume(m3)', 'q95', 'gasflowrateofmainspecies1022(es)']

    seperatrix_profiles = ['neseparatrixfromexpdata1019(m-3)']
    input_df = df_sep[main_engineer]
    target_df = df_sep[seperatrix_profiles]


    splits = [2.5, 2.5, 0.9, 0.315, 1.95, 15, 77.5, 3.7, 2.5]
    dict_splits = {key: value for key ,value in zip(main_engineer, splits)}

    matrix = []
    for key, item in dict_splits.items():
        datasplits = utils.create_joint_distr(input_df, target_df, col=key, split=item)
        new_coefs = main(datasplits)
        matrix.append(new_coefs.ravel())

    # print(matrix)
    plt.imshow(matrix)
    plt.xticks([i for i in range(0, len(main_engineer))], main_engineer)
    plt.yticks([i for i in range(0, len(main_engineer))], main_engineer)
    plt.colorbar()
    plt.show()
