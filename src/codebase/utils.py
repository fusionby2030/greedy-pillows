import numpy as np
import pandas as pd
import re


def convert_to_dataframe(data_loc, save_file=False):
    """
    :param data_loc: string
    :return: (pd dataframe)  returns the data as a dataframe.
    """
    shots = []
    with open(data_loc, 'r') as infile:
        headers = infile.readline()
        for line in infile:
            shot_n = line.strip()
            shot_n = shot_n.split(',')
            shot_n = [re.sub(r'[^a-zA-Z0-9_().-]', '', word) for word in shot_n]
            shots.append(shot_n)
    headers = headers.split(',')
    headers = [re.sub(r'[^a-zA-Z0-9_()-]', '', word) for word in headers]
    new_cols = []
    for i in range(0, len(headers)):
        new_cols.append(headers[i])
        if headers[i] not in no_errors:
            # print(headers[i])
            new_cols.append('error_' + headers[i])
    """
    Would like to convert the error columns to not those
    """
    df = pd.DataFrame(shots, columns=new_cols, dtype='float32')
    df = df.drop(3557)
    if save_file:
        df.to_csv(data_loc + '-dataframe')
    return df


def filter_dataframe(df, save_file=None):
    """
    parameters:
        df: pandas dataframe
        save_file: string if you want to save the file some location
    - Deuterium only
    - HRTS != 0.0
    - Remove kicks, RMPs and pellets
    - Keep impurities
    """
    df_filtered = df[(df['FLAGDEUTERIUM'] == 1.0) & (df['FLAGHRTSdatavalidated'] != 0.0) & (df['FLAGKicks'] == 0.0) & (df['FLAGRMP'] == 0.0)& (df['FLAGpellets'] == 0.0)]
    if save_file is not None:
        df_filtered.to_csv(save_file)
    return df_filtered

def prepare_dataset(file_loc, input_variables, target_variables, return_numpy=True):
    """
    splits dataset into X and y
    X contains the columns denoted in input_variables, and y the targets
    parameters:
        file_loc (string): location of the csv file to read

    returns input_df, target_df, input_err_df, target_err_df
    """
    df = pd.read_csv(file_loc)
    input_df = df[input_variables]
    target_df = df[target_variables]

    input_err_df = df[['error_' + col for col in input_variables]]
    target_err_df = df[['error_' + col for col in target_variables]]
    if return_numpy:
        return input_df.to_numpy(), target_df.to_numpy(), input_err_df.to_numpy(), target_err_df.to_numpy()
    return input_df, target_df, input_err_df, target_err_df

def plot_predictions(predictions, true_vals, color='orange', save_fig=False, save_dir='..../out/STANDALONE'):
    import matplotlib.pyplot as plt
    SMALL_SIZE = 22
    MEDIUM_SIZE = 30

    plt.rc('font', size=MEDIUM_SIZE, weight='bold')
    plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)
    fig = plt.figure(figsize=(18, 18))

    plt.scatter(true_vals, predictions, label='Predictions', c=color, edgecolors=(0, 0, 0))
    plt.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], 'r--')
    plt.xlabel('Real $n_e^{ped}$')
    plt.ylabel('Predicted $n_e^{ped}$')
    if save_fig and save_dir is not None:
        plt.savefig(save_dir)
    plt.show()

def create_joint_distr(X, y, col, split):
    """

    (a) self defined split of variable
    parameters:
        X: (array-like)
        y: (array-like)

    returns: X1, y1, X2, y2
    """
    all_index = X.index.to_list()
    X1 = X[X[col] > split]
    # Need to get indicies from X1 and pass to y1 and use the rest for the X2.
    indicies_X1 = X1.index.to_list()
    y1 = y.iloc[indicies_X1]
    inidices_x2 = np.setdiff1d(all_index, indicies_X1)
    X2 = X.iloc[inidices_x2]
    y2 = y.iloc[inidices_x2]
    return X1, y1, X2, y2
