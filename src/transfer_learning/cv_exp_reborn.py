# General
import argparse
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import List

# Networks
import torch
import torch.nn as nn
from peanuts.models.utils import set_module_torch, save_load_torch
from peanuts.models.torch_ensembles import AverageTorchRegressor
import numpy as np

# Data
from data import utils0 as utils
from sklearn.model_selection import RepeatedKFold

# Metrics
from sklearn.metrics import mean_squared_error

# Saving results
import pickle

"""
Define model like its done in Autoencoders
"""


class FFNN_transfer(nn.Module):
    def __init__(self, hidden_dims: List = None) -> None:
        super(FFNN_transfer, self).__init__()

        out_dim = 1
        input_dim = 10

        if hidden_dims is None:
            hidden_dims = [100, 100, 100]

        out_act = torch.nn.ReLU()

        last_dim = input_dim

        modules = []
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Linear(last_dim, h_dim), nn.LeakyReLU()))
            last_dim = h_dim

        self.layers = nn.Sequential(*modules)

        self.out = nn.Sequential(nn.Linear(last_dim, out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        out = self.out(y)
        return out


def train_model(model, X_train, y_train, learning_rate=0.001, batch_size=396, epochs=200, X_test=None, y_test=None, desc='Base'):
    # Set up data
    train_set = utils.ANNtorchdataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    if X_test is not None and y_test is not None:
        test_set = utils.ANNtorchdataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    else:
        test_loader = None
    if desc == 'Transfer':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler =  torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=0.000051, max_lr=0.081, cycle_momentum=False)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # we maybe want SGD for second optimizer!
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=0.0002, max_lr=0.0008, cycle_momentum=False)
    criterion = nn.MSELoss()

    best_mse = float("inf")

    training_iter = tqdm(range(epochs), position=2, leave=False, desc=desc)

    for epoch in training_iter:
        model.train()
        # batch_iter = tqdm(, position=4, leave=False, desc='training batch')
        for batch_idx, (inputs, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():
            training_iter.set_postfix(t_loss=loss.item())

        if test_loader:
            model.eval()
            with torch.no_grad():
                mse = 0.0
                for _, (inputs, target) in enumerate(test_loader):
                    output = model.forward(inputs)
                    mse += criterion(output, target)
                mse /= len(test_loader)
                if mse < best_mse:
                    best_mse = mse
                    best_state_dict = {"model_state_dict": model.state_dict()}
    if not test_loader:
        best_state_dict = {"model_state_dict": model.state_dict()}

    model.load_state_dict(best_state_dict['model_state_dict'])
    return model, best_state_dict


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        results = 0.0
        test_set = utils.ANNtorchdataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set))

        criteron = RMSELoss()

        for _, (inputs, targets) in enumerate(test_loader):
            output = model.forward(inputs)
            results = criteron(output, targets)
    return results


def freeze_layers(model, not_frozen_layer):
    for name, param in model.named_parameters():
        if param.requires_grad and not_frozen_layer in name:
            pass
        else:
            param.requires_grad = False
    return model



def main(**kwargs):
    dataset, input_scaler = utils.load_data_torch(neped_split=kwargs['neped_split'], n_samples=5)
    low_neped, high_neped, final_exam = dataset

    # X_valid, y_valid = torch.from_numpy(high_neped[0]), torch.from_numpy(high_neped[1][:, None])
    # X_exam, y_exam = torch.from_numpy(final_exam[0]), torch.from_numpy(final_exam[1][:, None])
    CV = RepeatedKFold(n_splits=4, n_repeats=2)

    cv_iterator_low = tqdm(CV.split(low_neped[0]), desc='CV', position=1, leave=True)

    results = {key: [] for key in np.concatenate((low_neped[1], high_neped[1], final_exam[1]))}

    cv_results = {'pre_low': [], 'pre_high': [], 'post_low': [], 'post_high': []}
    for (train, test) in cv_iterator_low:
        torch.manual_seed(42)

        X_train_low, y_train_low = low_neped[0][train], low_neped[1][train][:, None]
        X_test_low, y_test_low = torch.from_numpy(low_neped[0][test]), torch.from_numpy(low_neped[1][test])[:, None]

        # Initialize model
        model = FFNN_transfer(hidden_dims=args.hidden_layer_sizes, )

        # Train on low
        low_model, low_state = train_model(model, X_train_low, y_train_low, X_test=X_test_low, y_test=y_test_low, epochs=args.epochs_low, learning_rate=args.learning_rate_low, batch_size=args.batch_size_low)
        # low_model_cache = FFNN_transfer(hidden_dims=args.hidden_layer_sizes)
        # low_model_cache.load_state_dict(low_state['best_state_dict'])


        # Evaluate on Low validation
        low_results_low = evaluate_model(low_model, X_test_low, y_test_low)
        cv_results['pre_low'].append(low_results_low)

        # Transfer Learning
        # cv_iterator_high = tqdm(CV.split(high_neped[0]), position=2, leave=True)

        for (train, test) in CV.split(high_neped[0]):
            torch.manual_seed(42)
            X_train_high, y_train_high = high_neped[0][train], high_neped[1][train][:, None]
            X_test_high, y_test_high = high_neped[0][test], high_neped[1][test][:, None]
            low_model = freeze_layers(low_model, args.not_frozen_layer)

            model_transfer, high_state = train_model(low_model, X_train_high, y_train_high, desc='Transfer', X_test=X_test_high, y_test=y_test_high, epochs=args.epochs_high, learning_rate=args.learning_rate_high, batch_size=args.batch_size_high)

            # Evaluate transfer

            high_results_high = evaluate_model(model_transfer, X_test_high, y_test_high)
            high_results_low = evaluate_model(model_transfer, X_test_low, y_test_low)

            low_model = FFNN_transfer(hidden_dims=args.hidden_layer_sizes)
            low_model.load_state_dict(low_state['model_state_dict'])

            low_results_high = evaluate_model(low_model, np.vstack((X_train_high, X_test_high)), np.vstack((y_train_high, y_test_high)))

            cv_results['pre_high'].append(low_results_high)
            cv_results['post_high'].append(high_results_high)
            cv_results['post_low'].append(high_results_low)
            cv_iterator_low.set_postfix(low_low=low_results_low, low_high=low_results_high, post_high=high_results_high, post_low=high_results_low)

    return cv_results


def experiment(**kwargs):
    avg_losses = {'pre_low': [], 'pre_high': [], 'post_low': [], 'post_high': [],
                  'pre_low_std': [], 'pre_high_std': [], 'post_low_std': [], 'post_high_std': [], }

    exp_iterator = tqdm(np.linspace(4.5, 10, num=150), position=0, leave=True)
    for i in exp_iterator:
        kwargs['neped_split'] = i
        results = main(**kwargs)

        for key, value in results.items():
            avg_losses[key].append(np.mean(value))
            avg_losses[key + "_std"].append(np.std(value))

        exp_iterator.set_postfix(split=i)

    file_name = './out/transfer_learning_results_trial_Adam_cycle_' + str(len(args.hidden_layer_sizes)) + '_layer_freeze_' + args.not_frozen_layer +  '_layer.pickle'
    with open(file_name, 'wb') as file:
        pickle.dump(avg_losses, file)
        pickle.dump(kwargs, file)

    print(file_name)
    return avg_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs_l", '--batch_size_low', help='batch size during initial training on low neped', type=int,
                        default=396)
    parser.add_argument("-bs_h", '--batch_size_high', help='batch size during transfer training on high neped',
                        type=int, default=5)
    parser.add_argument('-lr_l', "--learning_rate_low", help='learning rate for initial training on low neped',
                        type=float, default=0.004)
    parser.add_argument('-lr_h', "--learning_rate_high", help='learning rate for transfer learning on high neped',
                        type=float, default=0.035)
    parser.add_argument('-ep_l', "--epochs_low", help='Number of epochs in low neped training (pre-transfer)', type=int,
                        default=200)
    parser.add_argument('-ep_h', "--epochs_high", help='Number of epochs in low neped training (pre-transfer)',
                        type=int, default=350)
    parser.add_argument('-ls', '--l_splits', help='Number of CV splits for low neped training (pre transfer)',
                        default=4, type=int)
    parser.add_argument('-hs', '--h_splits', help='Number of CV splits for high neped training (transfer)', default=4,
                        type=int)

    parser.add_argument('-hslist', '--hidden_layer_sizes', help='List of hidden layers [h1_size, h2_size, ..., ]',
                        nargs='+', default=[100, 100, 100, 100], type=int)
    parser.add_argument('-n_est', '--n_estimators', help='Number of ANNs in ensemble, 1 is default ANN', type=int,
                        default=1)
    parser.add_argument('-fl', '--freeze_layer', help='Which layer to NOT freeze', default='0.0', const='0.0',
                        nargs='?', choices=['5.0', '4.0', '3.0', '2.0', '1.0', '0.0', 'out'])
    parser.add_argument("-split", "--neped_split", help='Where to split data for highvs low neped', default=9.5,
                        type=float)
    parser.add_argument("-nfl", "--not_frozen_layer", help='Which layer not to freeze', default='out', type=str)
    args = parser.parse_args()

    config = vars(args)
    experiment(**config)
