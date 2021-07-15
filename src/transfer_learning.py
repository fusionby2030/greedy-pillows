"""
author: Adam Kit <adam.kit@aalto.fi>
date: 12.07.21

This program is supposed to be a rough outline
of the application of transfer learning in the prediction of high neped values
from JET pedestal database.

High neped is defined as nedped >= 0.95 x 10^{22}
At this density, a different regime of physics is occuring.
See B.Sc. repository for analysis for why the worry is being analysed.

Requirements:
    a) 3 "Datasets":
        - train_low_neped
        - train_high_neped
        - validation
            - combination of low and high neped
            - Currently at 15 shots a piece for low and high neped

    b) Torch network
        - Layer Variables:
            - trainable (learnable)
            - size
        - High level (reuse PedFFNN)
        - train found in AverageTorchRegressor

    c) stepwise training
        1. Train on low neped
        1.a Compare predictive capability on validation
        1.b freeze layers (save model?)

        2. train remaining trainable layers on high neped
        2.a compare predictive capability on validation
        2.b save model again

Should training be CV?
Probably not -> removing neped high kinda removes the need for CV .

METADATA TO STORE

model parameters
metrics
predictions
    - before transfer
    - after transfer


"""

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

# import itertools # used for the chain



from sklearn.metrics import mean_squared_error, mean_absolute_error # ye old metric gathers

from codebase.peanuts.models.utils import set_module_torch, save_load_torch # For saving and loading torch models
from codebase.peanuts.models.torch_ensembles import AverageTorchRegressor
from codebase.data.utils import ANNtorchdataset, load_data


import os
import pickle






def make_datasets(pickle_datasets=False):
    """
    Either take the df, or do everything here,
    I mean that sounds like a massive waste of CPU but why not, we don't care at the moment.
    """
    validation = train_low_neped = train_high_neped =  None
    df = pd.read_csv('/home/adam/data/seperatrix_dataset.csv')
    main_eng = ['Ip(MA)', 'B(T)', 'a(m)', 'averagetriangularity',
                 'P_NBI(MW)', 'P_ICRH(MW)','P_TOTPNBIPohmPICRH-Pshi(MW)',
                 'plasmavolume(m3)', 'q95', 'gasflowrateofmainspecies1022(es)']

    target = ['nepedheight1019(m-3)']

    # Scale main_eng params between 0 and 1
    # This is probably not best practice since I should split scaling
    # between train and validation, but fuck it this is proof of concept
    ss = StandardScaler()
    df[main_eng] = ss.fit_transform(df[main_eng]) # TODO: Warning about copying a slice from Dataframe.


    low_neped = df[df['nepedheight1019(m-3)'] < 9.5]
    high_neped = df[df['nepedheight1019(m-3)'] >= 9.5]

    # sample 15 shots from each set
    low_neped_sample = low_neped.sample(15, random_state=42)
    high_neped_sample = high_neped.sample(15, random_state=42)

    # remove those 15 shots from each set
    low_neped.drop(low_neped_sample.index, inplace=True)
    high_neped.drop(high_neped_sample.index, inplace=True)

    # combine removed samples to validation set
    df_sample = pd.concat([low_neped_sample, high_neped_sample])

    # now make into input and outputs set
    train_low_neped = ANNtorchdataset(low_neped[main_eng].to_numpy(np.float32), low_neped[target].to_numpy(np.float32))
    train_high_neped = ANNtorchdataset(high_neped[main_eng].to_numpy(np.float32), high_neped[target].to_numpy(np.float32))
    validation = ANNtorchdataset(df_sample[main_eng].to_numpy(np.float32), df_sample[target].to_numpy(np.float32))

    with open('./datasets.pickle', 'wb') as file:
        pickle.dump((train_low_neped, train_high_neped, validation), file)
    return train_low_neped, train_high_neped, validation


class PedFFNN(nn.Module):
    def __init__(self, **kwargs):
        super(PedFFNN, self).__init__()

        target_size = 1
        input_size = 10
        act_func = torch.nn.ELU()
        out_act = torch.nn.ReLU()

        last_size = input_size

        self.hidden_layers = torch.nn.ModuleList()
        hidden_layer_sizes = kwargs['hidden_layer_sizes']

        for size in hidden_layer_sizes:
            self.hidden_layers.append(self._fc_block(last_size, size, act_func))
            last_size = size

        self.out = self._fc_block(last_size, target_size, out_act)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)

        x = self.out(x)
        return x

    @staticmethod
    def _fc_block(in_c, out_c, act_func):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_c, out_c),
            act_func
        )
        return block

    def predict(self, X):
        self.eval()
        pred = None

        if isinstance(X, torch.Tensor):
            pred = self.forward(X)

        elif isinstance(X, np.ndarray):
            X = torch.Tensor(X)
            pred = self.forward(X)

        else:
            msg = 'The type of input to ensemble should be a torch.tensor or np.ndarray'
            raise ValueError(msg)

        return pred


def check_frozen_weights(model):
    # this is a future test case that I will totally write.
    print('Check if frozen actuallly')
    for name, para in model.named_parameters():
        if para.requires_grad is True:
            print(name)
            # para.requires_grad = False

def plot_comparison(true_vals, predictions_low, predictions_high, RMSE_dict = None, **args):
    import matplotlib.pyplot as plt
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

    fig, axs = plt.subplots(1, 1, figsize=(18, 18))
    axs.scatter(true_vals, predictions_low, s=100, label='Train on Low: {:.4}'.format(RMSE_dict['RMSE_low']))
    axs.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], 'r--')
    axs.scatter(true_vals, predictions_high, s=100, label='After Transfer: {:.4}'.format(RMSE_dict['RMSE']))
    axs.set(title='Transfer Learning on split $n_e^{ped} \geq 9.5 x 10^{21}$', xlabel='True $n_e^{ped} (10^{21}$m$^{-3})$', ylabel='Predicted')
    plt.legend()
    if args['output_loc']:
        file_name = args['output_loc'] + "transfer-learning_" + str(args['batch_size_transfer']) + "_" + str(args['non_freeze']) + "_" +str(args['learning_rate_transfer']) + "_" + '_'.join(str(e) for e in args['hidden_layer_sizes']) + '.png'
        # print(file_name)
        plt.savefig(file_name)
    else:
        plt.show()


def main(**kwargs):
    if os.path.exists(kwargs['dataset_loc']):
        with open(kwargs['dataset_loc'], 'rb') as file:
            datasets = pickle.load(file)
    else:
        datasets = make_datasets() # ((X_low_neped, y_low_neped), (X_high_neped, y_high_neped), (X_valid, y_valid))

    # print(datasets)

    low_neped_loader = torch.utils.data.DataLoader(datasets[0], kwargs['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets[2], 1, shuffle=True)
    base_model = PedFFNN
    model = AverageTorchRegressor(estimator=base_model, n_estimators=kwargs['n_estimators'], estimator_args={'hidden_layer_sizes': kwargs['hidden_layer_sizes']})
    model.set_optimizer('Adam', lr=kwargs['learning_rate'])
    model.double()

    # TODO: IF THERE EXISTS SOME CHECKPOINT ALREADY THEN SKIP
    state_low = model.fit(train_loader=low_neped_loader, test_loader=test_loader, epochs=kwargs['epochs'], cache_save=True)

    model = AverageTorchRegressor(estimator=base_model, n_estimators=kwargs['n_estimators'], estimator_args={'hidden_layer_sizes': kwargs['hidden_layer_sizes']})
    save_load_torch.load_cache_save(model, state_low)

    predictions_low = model.predict(datasets[2].inputs)
    RMSE_low = mean_squared_error(datasets[2].outputs, predictions_low, squared=False)

    # Freeze model
    for name, param in model.named_parameters():
        # print(name, param)
        if param.requires_grad and kwargs['non_freeze'] in name:
            pass
        else:
            param.requires_grad = False

    check_frozen_weights(model)
    # retrain on high neped
    high_neped_loader = torch.utils.data.DataLoader(datasets[1], kwargs['batch_size_transfer'], shuffle=True)
    model.set_optimizer('Adam', lr=kwargs['learning_rate_transfer'])


    state_high = model.fit(train_loader=high_neped_loader, test_loader=test_loader, epochs=kwargs['epochs_transfer'], cache_save=True)
    model = AverageTorchRegressor(estimator=base_model, n_estimators=kwargs['n_estimators'], estimator_args={'hidden_layer_sizes': kwargs['hidden_layer_sizes']})
    save_load_torch.load_cache_save(model, state_high)

    predictions_high = model.predict(datasets[2].inputs)
    RMSE_high = mean_squared_error(datasets[2].outputs, predictions_high, squared=False)
    print('\n')
    # print(list(itertools.chain(*predictions_high.tolist())))
    RMSE_dict = {'RMSE': RMSE_high.item(), 'RMSE_low': RMSE_low.item()}
    current_results = {'hyperparameters': kwargs, 'RMSE': RMSE_high.item(), 'RMSE_low': RMSE_low.item(), 'predictions_low': predictions_low.tolist(), 'predictions_high': predictions_high.tolist()}
    print(RMSE_dict)
    necessary_results = kwargs.copy()
    necessary_results['RMSE_post'] = RMSE_high
    necessary_results['RMSE_pre'] = RMSE_low
    necessary_results['hidden_layer_sizes'] = '_'.join(str(e) for e in args['hidden_layer_sizes'])
    # necessary_results = {k:[v] for k, v in necessary_results.items()}
    # print(necessary_results)
    # print(current_results)
    """
    df = pd.DataFrame.from_dict(results, dtype=object)
    print(df)
    df.to_csv(kwargs['output_file'], mode='a', header=not os.path.exists(kwargs['output_file']))
    """
    if kwargs['plot'] >=1:
        plot_comparison(datasets[2].outputs, predictions_low.tolist(), predictions_high.tolist(), RMSE_dict, **kwargs)
        try:
            write_to_file(necessary_results, kwargs['output_loc'] + 'results_transfer_search.csv')
            print('updated CSV I suppose')
        except Exception as e:
            print('something bad happened so making a new file')
            print(e)
            create_file(necessary_results,kwargs['output_loc'] + 'results_transfer_search.csv')
    return current_results

def write_to_file(dict_to_write, output_loc):
    from csv import DictWriter
    if not os.path.exists(output_loc):
        raise Exception
    headers = list(dict_to_write.keys())
    with open(output_loc, 'a', newline='') as f_object:
        dict_writer_object = DictWriter(f_object, fieldnames=headers)
        dict_writer_object.writerow(dict_to_write)
        f_object.close()

def create_file(dict_to_write, output_loc):
    import csv
    necessary_results = {k:[v] for k, v in dict_to_write.items()}
    df_old = pd.DataFrame.from_dict(necessary_results)
    df_old.to_csv(output_loc, index=False)

import argparse
if __name__ == '__main__':
    torch.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", help='batch size during training/validation', type=int, default=396)
    parser.add_argument("-bst", '--batch_size_transfer', help='batch size during transfer training', type=int, default=1)
    parser.add_argument("-ep", "--epochs", help='epochs for initial pre training', type=int, default=200)
    parser.add_argument("-ept", "--epochs_transfer", help='epochs for initial pre training', type=int, default=200)
    parser.add_argument('-lr', "--learning_rate", help='learning rate', type=float, default=0.004)
    parser.add_argument('-lrt', "--learning_rate_transfer", help='learning rate for transfer learning', type=float, default=0.00001)
    parser.add_argument('-n_splits', help='number of folds in CV', type=int, default=5)
    parser.add_argument('-n_repeats', help='number of repeats of CV', type=int, default=2)
    parser.add_argument('-n_estimators', help='Number of ANNs in ensemble, 1 is default ANN',type=int, default=1)
    parser.add_argument('-plot', help='Include plot at the end', action="count", default=0)
    parser.add_argument('-dataset_loc', help='If the dataset is a pickle, then load it like this', type=str, default='./datasets.pickle')
    parser.add_argument('-output_loc', help='Which csv will hold your output', type=str, default='')
    parser.add_argument('-non_freeze', help='what layer NOT to freeze', default='out', const='out', nargs='?', choices=['5.0', '4.0', '3.0', '2.0', 'out'])
    parser.add_argument('-st', '--smoke_test', help='Smoke Test, quickly check if it works', action="count", default=0)
    parser.add_argument('-hslist', '--hidden_layer_sizes', help='List of hidden layers [h1_size, h2_size, ..., ]', nargs='+', default=[600, 600, 600, 600], type=int)
    args_namespace = parser.parse_args()
    args = vars(args_namespace)

    # args['hidden_layer_sizes'] = [636, 537, 295, 261]
    if args['smoke_test'] >=1:
        smoke_args = {'hidden_layer_sizes': [10, 10, 10, 10], 'n_estimators': 1, 'epochs': 25}
        args.update(smoke_args)
        print('Starting smoke test')
        main(**args)
        print('Smoke Test Passed')
    else:
        main(**args)
