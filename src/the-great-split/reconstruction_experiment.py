"""
Goal: train AE or VAE on low neped data, then see if reconstruction error for high neped is large enough to detect.
"""

# General
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Networks
from models import *
import torch

# Data
from data import utils0 as utils
from sklearn.model_selection import RepeatedKFold
# Metrics
from sklearn.metrics import mean_squared_error

# Saving results
import pickle


def train_model(X_train, y_train, model_type):
    model_params = {"input_dim": 10, "hidden_dims": args.hidden_dims, "latent_dim": args.latent_dim, "cond_dim": 1}
    if model_type not in ['VanillaVAE', 'BetaVAE', 'ConditionalVAE', 'SimpleAE']:
        print(vae_models.values())
        raise ValueError('Invalid Model type, choose VanillaVAE, BetaVAE, ConditionalVAE, or SimpleAE')
    else:
        model = vae_models[model_type](**model_params)

    train_set = utils.ANNtorchdataset(X_train, y_train)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=True)

    # Initalize Model

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Train
    model.train()
    auto_train = tqdm(range(args.epochs), desc='Autoencoder Training', position=2, leave=False)
    for epoch in auto_train:
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            results = model.forward(x, conditions=y)
            train_loss = model.loss_function(*results, M_N=args.batch_size)
            train_loss['loss'].backward()
            optimizer.step()
            # print(train_loss)
            # print('\n')
            # print(train_loss['loss'].item())
        # print(f'Epoch {epoch}\n')
        auto_train.set_postfix(loss=train_loss['loss'].item(), score=-np.log10(train_loss['loss'].item()))
    return model


def main(**kwargs):
    dataset, input_scaler = utils.load_data_torch(neped_split=kwargs['neped_split'], n_samples=5)
    low_neped, high_neped, final_exam = dataset

    X_valid, y_valid = torch.from_numpy(high_neped[0]), torch.from_numpy(high_neped[1][:, None])
    X_exam, y_exam = torch.from_numpy(final_exam[0]), torch.from_numpy(final_exam[1][:, None])
    CV = RepeatedKFold(n_splits=5, n_repeats=5)

    cv_iterator = tqdm(CV.split(low_neped[0]), desc='CV', position=1, leave=False)

    results = {key: [] for key in np.concatenate((low_neped[1], high_neped[1], final_exam[1]))}

    avg_losses = {'low': [], 'high': []}

    for (train, test) in cv_iterator:
        X_train, y_train = low_neped[0][train], low_neped[1][train]
        X_test, y_test = torch.from_numpy(low_neped[0][test]), torch.from_numpy(low_neped[1][test][:, None])

        y_train = y_train[:, None]

        model = train_model(X_train, y_train, model_type=args.vae_type)

        model.eval()
        # all_results = {}

        losses = []
        for x, y in zip(X_test, y_test):
            predictions = model.generate(x, conditions=y)
            loss = mean_squared_error(y_true=x, y_pred=predictions.detach())
            if isinstance(y, torch.Tensor):
                y = y.item()
            results[y].append(loss.item())
            losses.append(loss)
        avg_losses['low'].append(np.mean(losses))

        losses = []
        for x, y in zip(X_valid, y_valid):
            predictions = model.generate(x, conditions=y)
            loss = mean_squared_error(y_true=x, y_pred=predictions.detach())
            if isinstance(y, torch.Tensor):
                y = y.item()
            results[y].append(loss.item())
            losses.append(loss)
        avg_losses['high'].append(np.mean(losses))

        for x, y in zip(X_exam, y_exam):
            predictions = model.generate(x, conditions=y)
            loss = mean_squared_error(y_true=x, y_pred=predictions.detach())
            if isinstance(y, torch.Tensor):
                y = y.item()
            results[y].append(loss.item())
            losses.append(loss)

        cv_iterator.set_postfix(high=avg_losses['high'][-1], low=avg_losses['low'][-1])

    avg_losses['low_std'] = np.std(avg_losses['low'])
    avg_losses['low'] = np.mean(avg_losses['low'])
    avg_losses['high_std'] = np.std(avg_losses['high'])
    avg_losses['high'] = np.mean(avg_losses['high'])

    end_results = {'y_val': [], 'std': [], 'recon': []}
    for key, value in results.items():
        end_results['y_val'].append(key)
        end_results['std'].append(np.std(value))
        end_results['recon'].append(np.mean(value))

    return end_results, avg_losses


def experiment(**kwargs):
    final_results = {}
    final_losses = {'split': [], 'low': [], 'high': [], 'low_std': [], 'high_std': []}

    exp_iterator = tqdm(np.linspace(5, 10, num=50), position=0)
    for i in exp_iterator:
        kwargs['neped_split'] = i
        results, avg_losses = main(**kwargs)
        final_results[str(i)] = results
        final_losses['split'].append(i)
        final_losses['low'].append(avg_losses['low'])
        final_losses['high'].append(avg_losses['high'])
        final_losses['low_std'].append(avg_losses['low_std'])
        final_losses['high_std'].append(avg_losses['high_std'])
        exp_iterator.set_postfix(split=i)

    file_name = './out/anom_detect_' + args.vae_type + '_ld' + str(args.latent_dim) + '_split5to10' + '_results.pickle'
    with open(file_name, 'wb') as file:
        pickle.dump(final_results, file)
        pickle.dump(final_losses, file)

    fig5 = plt.figure()

    plt.scatter(final_losses['split'], final_losses['low'], label='low')
    plt.scatter(final_losses['split'], final_losses['high'], label='high')
    plt.xlabel('neped split')
    plt.ylabel('Recon Loss')
    plt.legend()

    digitized = np.digitize(results['y_val'], np.linspace(1, 12, num=12))
    residual = [np.abs(results['recon'])[digitized == i].mean() for i in range(1, 12)]

    fig = plt.figure()

    plt.bar(np.linspace(1, 12, 12)[1:12], residual)

    fig2 = plt.figure()
    plt.scatter(results['y_val'], results['recon'])
    plt.errorbar(results['y_val'], results['recon'], fmt='none', yerr=results['std'], alpha=0.3, color='grey')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruction error for autoencoder, i.e., anomoly detection')
    parser.add_argument("-bs", "--batch_size", help='batch size for training', type=int, default=419)
    parser.add_argument("-ep", "--epochs", help='num epochs', type=int, default=250)
    parser.add_argument("-lr", "--learning_rate", help='learning rate', type=float, default=0.00039)
    parser.add_argument("-ld", "--latent_dim", help='Latent Dimensions of AE', default=50, type=int)
    parser.add_argument('-hslist', '--hidden_dims', help='List of hidden layers [h1_size, h2_size, ..., ]', nargs='+',
                        default=[50, 50, 50], type=int)
    parser.add_argument("-vae", "--vae_type", help='Which VAE to use, choose SimpleAE if you do not know',
                        default='SimpleAE', type=str)

    parser.add_argument("-split", "--neped_split", help='Where to split data for highvs low neped', default=9.5,
                        type=float)

    args = parser.parse_args()
    config = vars(args)
    experiment(**config)
