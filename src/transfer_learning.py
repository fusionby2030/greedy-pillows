"""
author: Adam Kit <adam.kit@aalto.fi>
date: 12.07.21


Various variable names occur, which are cleared up here.
The model that is trained on low neped values is sometimes refered to as:
    - young model
        - Is fed the low_neped_dataloader, or young_loader
        - makes young_predictions, which have a RMSE_low or RMSE_young compared to the true values
The model that is trained on high neped values via transfer learning of the young model is:
    - transfer_model
        - Is fed wise_loader
        - makes transfere_predictions which have RMSE_high or RMSE_old in comparison to true neped values



TODO:
    - A more robust CV method, so that it is not just dependent on the data
"""

import pandas as pd # we will use this for saving the data at the end
import numpy as np # I don't think we even use this but I can sleep better knowing numpy is there for me

import torch
import torch.nn as nn

from sklearn.metrics import mean_squared_error # ye old metric gathers

from codebase.peanuts.models.utils import set_module_torch, save_load_torch # For saving and loading torch models
from codebase.peanuts.models.torch_ensembles import AverageTorchRegressor # The 'make everything look easy with the .fit method' class
from codebase.data.utils import ANNtorchdataset, load_data_torch # loading the data

import os

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


def make_dataloaders(datasets, batch_size, batch_size_transfer):
    """
    This is to make the different dataloaders for training the network
    dataloaders are a way of 'feeding' the data quickly to a ANN. Each library
    typically has its own type of dataloading class, which for pytorch is:
    torch.utils.data.DataLoader()

    - datasets (tuple, size: 3, type: Torch dataset): see the load_data_torch() method in codebase.data.utils
    - batch_size (int): size of mini-batches that are given during the training
        of the pre-transfer network,
        i.e., mini-batch size of datset of low neped values
    - batch_size_transfer (int): same as above but for the dataloader of transfer net

    returns:
        three loaders, one for trianing the 'young' model with low_neped,
            - low_neped_loader: see name
            - high_neped_loader: see name
            - final_exam_loader: the validation loader

    """
    low_neped_loader = torch.utils.data.DataLoader(datasets[0], batch_size, shuffle=True)
    high_neped_loader = torch.utils.data.DataLoader(datasets[1], batch_size_transfer, shuffle=True)
    final_exam_loader = torch.utils.data.DataLoader(datasets[2], 1, shuffle=True)
    return low_neped_loader, high_neped_loader, final_exam_loader

def prepare_model(base_model, n_estimators, hidden_layer_sizes, optimizer, learning_rate, post_train=False, transfer=False):
    # So here we have to initalize the model
    # If it is already trained, then we just create a blank copy of the ANN (Average torch regressor)
    # And load it with the saved version
    # The transfer redirects the load to a different directory, the transfer models directory.

    if post_train:
        model = AverageTorchRegressor(estimator=base_model, n_estimators=n_estimators, estimator_args={'hidden_layer_sizes': hidden_layer_sizes})
        if transfer:
            save_load_torch.load(model, save_dir='./out/transfer_models/')
        else:
            save_load_torch.load(model)
    else:
        model = AverageTorchRegressor(estimator=base_model, n_estimators=n_estimators, estimator_args={'hidden_layer_sizes': hidden_layer_sizes})
        model.set_optimizer(optimizer, lr=learning_rate)
        model.double()
    return model

def fit_model(model, train_load, test_load, epochs, transfer=False):
    """
    Fit a model, and return the trained best version of the model.
    The reason this is possible is because the fitting funciton saves the
    best performing epoch of the model (best perf. on validation set)
    """
    if transfer:
        model.fit(train_load, epochs, test_loader=test_load, save_dir='./out/transfer_models/')
        # trained_model = save_load_torch.load(model, savedir='./out/transfer_models/')
    else:
        # model.fit(train_load, epochs, test_loader=test_load)
        # There is an error when loading previous checkpoint
        # When the previous model is diffefrent than the last trained
        # Need to maybe add date to the string.
        # Or delete the previous version before
        # Or make a try catch
        # So many solutions!
        if os.path.exists('./AverageTorchRegressor_PedFFNN_1_ckpt.pth'):
            print('pre-trained model found, using it as base model')
            trained_model = save_load_torch.load(model)
        else:
            print('No pre-trained model found, refitting base model')
            model.fit(train_load, epochs, test_loader=test_load)
            # trained_model = save_load_torch.load(model)

def get_predictions(model, X):
    # Pretty self explanatory
    return model.predict(X)

def freeze_layers(model, non_freeze):
    # Here we freeze all layers except the one denoted by non_freeze
    # Non freeze is a string, which corresponds to a number 0.0, 1.0, -> num layers -1
    # So if we want to freeze layer 4 of a 4 layer network, non_freeze is '3.0'
    for name, param in model.named_parameters():
        if param.requires_grad and non_freeze in name:
            # skip the layer you don't want to freeze
            pass
        else:
            # this is freezing, i.e., removing gradient in the backward pass
            param.requires_grad = False
    return model

def prepare_transfer_model(model, optimizer=None, learning_rate=None):
    model.set_optimizer(optimizer, lr=learning_rate)


def get_metrics(young_predictions, transfer_predictions, true_vals):
    RMSE_young = mean_squared_error(y_true=true_vals, y_pred=young_predictions, squared=False)
    RMSE_old = mean_squared_error(y_true=true_vals, y_pred=transfer_predictions, squared=False)
    out_array = np.column_stack((true_vals, young_predictions.numpy(), transfer_predictions.numpy()))
    df = pd.DataFrame(out_array, columns=['true', 'young', 'transfer'])
    df.to_csv('./out/transfer_models/predictions.csv') # so we can do further plotting later!
    RMSE_dict = {'RMSE_low': RMSE_young, 'RMSE_high':RMSE_old}
    return RMSE_dict

def main(**kwargs):
    # First step is always data prep
    datasets = load_data_torch()
    young_loader, wise_loader, final_exam_loader = make_dataloaders(datasets, kwargs['batch_size'], kwargs['batch_size_transfer'])
    # Initialize our first 'young' model and fit it
    young_model = prepare_model(PedFFNN, kwargs['n_estimators'], kwargs['hidden_layer_sizes'], kwargs['optimizer'], kwargs['learning_rate'])
    fit_model(young_model, young_loader, final_exam_loader, epochs=kwargs['epochs'])
    # We need to reload the best version of the model that is saved at ./AverageTorchRegressor_PedFFNN_1_ckpt.pth
    trained_young_model = prepare_model(PedFFNN, kwargs['n_estimators'], kwargs['hidden_layer_sizes'], kwargs['optimizer'], kwargs['learning_rate'], post_train=True)
    young_predictions = get_predictions(trained_young_model, datasets[2].inputs)
    #
    model_transfer = freeze_layers(trained_young_model, kwargs['non_freeze'])

    prepare_transfer_model(model_transfer, kwargs['optimizer'], kwargs['learning_rate_transfer'])
    fit_model(model_transfer, wise_loader, final_exam_loader, epochs=kwargs['epochs_transfer'], transfer=True)
    trained_transfer_model = prepare_model(PedFFNN, kwargs['n_estimators'], kwargs['hidden_layer_sizes'], kwargs['optimizer'], kwargs['learning_rate'], post_train=True, transfer=True)
    transfer_predictions = get_predictions(trained_transfer_model, datasets[2].inputs)

    RMSE = get_metrics(young_predictions, transfer_predictions, datasets[2].outputs)
    print(RMSE)
    plot_comparison(datasets[2].outputs, young_predictions, transfer_predictions, RMSE)


def plot_comparison(true_vals, predictions_low, predictions_high, RMSE_dict = None, output_loc=None):
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
    axs.scatter(true_vals, predictions_high, s=100, label='After Transfer: {:.4}'.format(RMSE_dict['RMSE_high']))
    axs.set(title='Transfer Learning on split $n_e^{ped} \geq 9.5 x 10^{21}$', xlabel='True $n_e^{ped} (10^{21}$m$^{-3})$', ylabel='Predicted')
    plt.legend()
    plt.show()
    if output_loc is not None:
        file_name = output_loc + "transfer-learning_trial"  + '.png'
        # print(file_name)
        plt.savefig(file_name)
    else:
        plt.show()


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
    parser.add_argument('-optim', '--optimizer', help='Which Optimizer to use during pretrainng', default='Adam', type=str)
    args_namespace = parser.parse_args()
    args = vars(args_namespace)
    main(**args)
    # args['hidden_layer_sizes'] = [636, 537, 295, 261]
    # TODO: Add a few test cases
