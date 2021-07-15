from sacred import Experiment
from sacred.observers import FileStorageObserver

import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn

from codebase.peanuts.models.utils import set_module_torch, save_load_torch
from codebase.peanuts.models.torch_ensembles import AverageTorchRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error
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
# Experiment Instance

# Import the dataset ingredient

from codebase.data.sacred_data import data_ingredient, load_data, ANNtorchdataset

ex = Experiment('transfer_learning', ingredients=[data_ingredient])
ex.observers.append(FileStorageObserver('./out/sacred_transfer_modeling'))

# Experiment configuration
@ex.config
def my_config():
    batch_size_transfer = 8
    non_freeze = '3.0'
    learning_rate_transfer = 0.00006
    epochs_transfer = 200
    epochs = 200
    batch_size = 396

    optimizer = 'Adam'
    optimizer_transfer = 'Adam'
    learning_rate = 0.004

    n_estimators = 1
    base_model = PedFFNN
    hidden_layer_sizes = [400, 400, 400, 400]

@ex.named_config
def smoke_test():
    batch_size_transfer = 8
    non_freeze = '3.0'
    learning_rate_transfer = 0.00006
    epochs_transfer = 15
    epochs = 15
    batch_size = 396

    optimizer = 'Adam'
    optimizer_transfer = 'Adam'
    learning_rate = 0.004

    n_estimators = 1
    base_model = PedFFNN
    hidden_layer_sizes = [10, 10, 10, 10]

@ex.capture
def make_torch_dataloaders(datasets, batch_size, batch_size_transfer):
    low_neped_loader = torch.utils.data.DataLoader(datasets[0], batch_size, shuffle=True)
    high_neped_loader = torch.utils.data.DataLoader(datasets[1], batch_size_transfer, shuffle=True)
    final_exam_loader = torch.utils.data.DataLoader(datasets[2], 1, shuffle=True)
    return low_neped_loader, high_neped_loader, final_exam_loader

# Capture functions take config values and do something with them
@ex.capture
def prepare_young_model(base_model, n_estimators, hidden_layer_sizes, optimizer, learning_rate, post_train=False):
    if post_train:
        model = AverageTorchRegressor(estimator=base_model, n_estimators=n_estimators, estimator_args={'hidden_layer_sizes': hidden_layer_sizes})
        save_load_torch.load(model)
    else:
        model = AverageTorchRegressor(estimator=base_model, n_estimators=n_estimators, estimator_args={'hidden_layer_sizes': hidden_layer_sizes})
        model.set_optimizer(optimizer, lr=learning_rate)
        model.double()
    return model


@ex.capture
def fit_young_model(model, train_loader, test_loader, epochs, _log):
    if os.path.exists('./AverageTorchRegressor_PedFFNN_1_ckpt.pth'):
        _log.warning('Using last known checkpoint from: ./AverageTorchRegressor_PedFFNN_1_ckpt.pth \nif you think this is wrong, delete whatever checkpoint exists in the dir,\nrerun file and the model will fit again')
    else:
        model.fit(train_loader, epochs, test_loader=test_loader)

@ex.capture
def freeze_layers(model, non_freeze):
    for name, param in model.named_parameters():
        if param.requires_grad and non_freeze in name:
            pass
        else:
            param.requires_grad = False
    return model

@ex.capture
def prepare_transfer_model(model, optimizer_transfer, learning_rate_transfer):
    model.set_optimizer(optimizer_transfer, lr=learning_rate_transfer)

@ex.capture
def fit_transfer_model(model, train_loader, test_loader, epochs_transfer):
    model.fit(train_loader, epochs_transfer, test_loader=test_loader, save_dir='./out/transfer_models/')
    ex.add_artifact('./out/transfer_models/AverageTorchRegressor_PedFFNN_1_ckpt.pth', name='final_trained_transfer_model.pth')
    
@ex.capture
def load_fitted_transfer(base_model, n_estimators, hidden_layer_sizes):
    model = AverageTorchRegressor(estimator=base_model, n_estimators=n_estimators, estimator_args={'hidden_layer_sizes': hidden_layer_sizes})
    save_load_torch.load(model, './out/transfer_models/')
    return model

@ex.capture
def get_predictions(model, predictive_inputs):
    return model.predict(predictive_inputs)

@ex.capture
def log_metrics(_run, young_predictions, transfer_predictions, true_vals):
    RMSE_young = mean_squared_error(y_true=true_vals, y_pred=young_predictions, squared=False)
    RMSE_old = mean_squared_error(y_true=true_vals, y_pred=transfer_predictions, squared=False)
    _run.log_scalar('young_RMSE', RMSE_young)
    _run.log_scalar('old_RMSE', RMSE_old)
    out_array = np.column_stack((true_vals, young_predictions.numpy(), transfer_predictions.numpy()))
    df = pd.DataFrame(out_array, columns=['true', 'young', 'transfer'])
    df.to_csv('./out/transfer_models/predictions.csv')
    ex.add_artifact('./out/transfer_models/predictions.csv')

# The main running file decorated with @ex.automain
@ex.automain
def main():
    datasets = load_data()
    low_neped_loader, high_neped_loader, final_exam_loader = make_torch_dataloaders(datasets)
    model = prepare_young_model()
    fit_young_model(model, low_neped_loader, final_exam_loader)
    model_young = prepare_young_model(post_train=True)
    young_predictions = get_predictions(model_young, datasets[2].inputs)
    model_transfer = freeze_layers(model_young)
    prepare_transfer_model(model_transfer)
    fit_transfer_model(model_transfer, high_neped_loader, final_exam_loader)
    final_model = load_fitted_transfer()
    transfer_predictions = get_predictions(final_model, datasets[2].inputs)
    log_metrics(young_predictions=young_predictions, transfer_predictions=transfer_predictions, true_vals=datasets[2].outputs)
