"""
author: Adam Kit <adamkit11@gmail.com>

Goal: Handle both High and Low neped using latent space of autoencoder


- Datasplits
    - Low
    - High
    - Low & High neped (concat above two)
    - 15 points from high and low for final exam set
    -

- training
    - Autoencoder (AE)
        - Simple Vanilla AE
    - classifier from latent space of AE (CLF)
        - Nearest Neighbors seems reasonable from SKLEARN
    - Low neped model (LnM)
        1. NN
    - high neped model (HnM)
        1. NN

- Flow of Information
    - Inputs -> Autoencoder
    - Encode-> Latent Space
    - Latent Space-> Classifier
    - Classifier -> low or high model
"""

import torch
from data import utils0 as utils
from models import *
from tqdm import tqdm

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from matplotlib.colors import ListedColormap
SMALL_SIZE = 40
MEDIUM_SIZE = 45
BIGGER_SIZE = 50

"""plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
"""
def plot_latent_tsne(datasets, autoencoder):
    X_train = np.concatenate((datasets[0][0], datasets[1][0]), axis=0)
    y_train = np.concatenate((datasets[0][1], datasets[1][1]), axis=0)
    X_test = datasets[2][0]
    y_test = datasets[2][1]
    # Combine into a torch dataset
    train_set = utils.ANNtorchdataset(X_train, y_train)
    test_set = utils.ANNtorchdataset(X_test, y_test)
    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_set, len(train_set), shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, len(test_set), shuffle=True)

    for i, (x, y) in enumerate(train_loader):
        z = autoencoder.encode(x)
        if type(z) == list:
            mu, logvar = z
            z = autoencoder.reparameterize(mu, logvar).detach()
        else:
            z = z.detach()
        Z_embed = TSNE(n_components=2).fit_transform(z)
        plt.scatter(Z_embed[:, 0], Z_embed[:, 1], c=y, cmap="Spectral", s=400)

    plt.title('TSNE Latent Space')
    plt.colorbar()
    plt.show()


def train_autoencoder(datasets, model_type: str = 'SimpleAE', **kwargs):
    # Concatenate data
    X_train = np.concatenate((datasets[0][0], datasets[1][0]), axis=0)
    y_train = np.concatenate((datasets[0][1], datasets[1][1]), axis=0)
    X_test = datasets[2][0]
    y_test = datasets[2][1]
    # Combine into a torch dataset
    train_set = utils.ANNtorchdataset(X_train, y_train)
    test_set = utils.ANNtorchdataset(X_test, y_test)
    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, args.batch_size, shuffle=True)

    # Initalize Model
    model_params = {"input_dim": 10, "hidden_dims": args.hidden_dims, "latent_dim": args.latent_dim}
    model = vae_models[model_type](**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['learning_rate'])
    # Train

    auto_train = tqdm(range(args.epochs), desc='Autoencoder Training')
    for epoch in auto_train:
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            results = model.forward(x, conditions=y)

            train_loss = model.loss_function(*results, M_N = args.batch_size)
            train_loss['loss'].backward()
            optimizer.step()
            # print(train_loss)
            #print('\n')
            # print(train_loss['loss'].item())
        # print(f'Epoch {epoch}\n')
        auto_train.set_postfix(loss = train_loss['loss'].item(), score=np.log10(train_loss['loss'].item()))

    # model.plot_latent_space(train_loader)
    # model.plot_latent_space(test_loader)
    return model

def train_classifier(dataset, autoencoder, **kwargs):
    autoencoder.eval()
    datasets = dataset
    low_neped, high_neped = datasets[0], datasets[1]

    X_train = np.concatenate((low_neped[0].copy(), high_neped[0].copy()), axis=0)
    y_train = np.concatenate((np.zeros_like(low_neped[1]),np.ones_like(high_neped[1])))

    X_test, y_test = datasets[2][0].copy(), datasets[2][1].copy()
    y_test[y_test < 9.5] = 0
    y_test[y_test >= 9.5] = 1

    X_train_latent = autoencoder.encode(torch.from_numpy(X_train))
    X_test_latent = autoencoder.encode(torch.from_numpy(X_test))
    if type(X_train_latent) == list:
        mu, logvar = X_train_latent
        X_train_latent = autoencoder.reparameterize(mu, logvar).detach()
        mu, logvar = X_test_latent
        X_test_latent = autoencoder.reparameterize(mu, logvar).detach()
    else:
        X_train_latent = X_train_latent.detach()
        X_test_latent = X_test_latent.detach()


    # clf = KNeighborsClassifier(n_neighbors=2, weights='distance')
    clf = NearestCentroid(shrink_threshold=0.24, metric='cosine')
    # clf = KMeans(n_clusters=2, random_state=42)
    clf.fit(X_train_latent, y_train)

    y_preds = clf.predict(X_test_latent)
    y_preds_trian = clf.predict(X_train_latent)


    acc_train = accuracy_score(y_train, y_preds_trian)
    acc_test = accuracy_score(y_test, y_preds)

    f1_test = f1_score(y_test, y_preds)
    f1_train = f1_score(y_train, y_preds_trian)

    print('\n # Trained Classifier Final Exam, ACC {:.4}, f1 {:.4}'.format(acc_test, f1_test))
    print('\n # Trained Classifier Training, ACC {:.4}, f1 {:.4}'.format(acc_train, f1_train))

    return clf
def train_regressor(dataset, model_type = 'RF'):
    datasets = dataset
    X_train, y_train = datasets[0][0], datasets[0][1]
    X_test, y_test = datasets[1][0], datasets[1][1]

    if model_type == 'RF':
        reg = RandomForestRegressor(random_state=42)
    else:
        kernel = WhiteKernel() + RationalQuadratic()
        reg = GaussianProcessRegressor(kernel=kernel)
    reg.fit(X_train, y_train)

    y_preds = reg.predict(X_test)
    y_preds_train = reg.predict(X_train)

    # print(y_test, y_preds)
    # print(y_train, y_preds_train)


    RMSE_test = mean_squared_error(y_test, y_preds, squared=False)
    RMSE_train = mean_squared_error(y_train, y_preds_train, squared=False)

    print('\n # Trained Regressor, RMSE_test {:.4}, RMSE_train {:.4}'.format(RMSE_test, RMSE_train))
    return reg
    """
    plt.scatter(y_test, y_preds)
    plt.scatter(y_train, y_preds_train)
    plt.plot([min(y_train), min(y_train)], [max(y_test), max(y_test)])
    plt.show()
    """

def make_pretty_plot(datasets, autoencoder, classifier, low_model, high_model):

    # Autoencoder plot (TSNE or just hidden layer stuffs?)
    AE_X_train = np.concatenate((datasets[0][0], datasets[1][0]), axis=0)
    AE_y_train = np.concatenate((datasets[0][1], datasets[1][1]), axis=0)
    X_test, y_test = datasets[2][0].copy(), datasets[2][1].copy()
    y_labels = y_test.copy()
    y_labels[y_labels < 9.5] = 0
    y_labels[y_labels >= 9.5] = 1

    train_set = utils.ANNtorchdataset(AE_X_train, AE_y_train)
    test_set = utils.ANNtorchdataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_set, len(train_set), shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, len(test_set), shuffle=True)
    autoencoder.eval()

    autoencoder.plot_latent_space(train_loader)
    fig, axs = plt.subplots(2, 2, dpi=500)
    axs = axs.ravel()
    for _, (x, y) in enumerate(train_loader):
        X_train_latent = autoencoder.encode(x).detach()
        Z_embed = TSNE(n_components=2).fit_transform(X_train_latent)
        axs[0].scatter(Z_embed[:, 0], Z_embed[:, 1], c=y, cmap="Spectral", s=400)
    for _, (x, y) in enumerate(test_loader):
        X_test_latent = autoencoder.encode(x).detach()

    # Classifier plot
    clf = classifier
    y_preds = clf.predict(X_test_latent)
    h = 1
    X = X_test_latent
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['salmon', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'darkblue'])

    axs[1].pcolormesh(xx, yy, Z, cmap=cmap_light)
    axs[1].scatter(X[:, 0], X[:, 1], c=y_labels, cmap=cmap_bold)

    # Regression Plot

    for input, target, label in zip(X_test, y_test, y_labels):
        latent_rep = autoencoder.encode(torch.from_numpy(input)).detach()
        # latent_rep = np.array(latent_rep, dtype=np.float32)
        classified_as = classifier.predict(latent_rep.reshape(1, -1))[0]
        print('\n##### New Target ####')
        print('CLF PRED {}, Actual {}'.format(classified_as, label))
        if classified_as == 0:
            print('------> Feeding to Low Model')
            y_pred = low_model.predict(input.reshape(1, -1))
            axs[2].scatter(target, y_pred, s=400, c='salmon', edgecolors=(0,0,0))
        else:
            print('------> Feeding to High Model')
            y_pred = high_model.predict(input.reshape(1, -1))
            axs[2].scatter(target, y_pred, s=400, c='cornflowerblue', edgecolors=(0,0,0))

        print('Predicted {}, Actual {}'.format(y_pred, target))

    # Some final regressor  TBD
    plt.tight_layout()
    plt.show()
    pass

def take_final_exam(dataset, autoencoder, classifier, low_model, high_model):
    autoencoder.float()
    fig = plt.figure(figsize=(18, 18))
    final_exam_inputs = dataset[0]
    final_exam_targets = dataset[1]
    final_exam_labels = dataset[1].copy()
    final_exam_labels[final_exam_labels < 9.5] = 0
    final_exam_labels[final_exam_labels >= 9.5] = 1

    plt.plot([min(final_exam_targets), max(final_exam_targets)], [min(final_exam_targets), max(final_exam_targets)], c='black')


    for input, target, label in zip(final_exam_inputs, final_exam_targets, final_exam_labels):
        latent_rep = autoencoder.encode(torch.from_numpy(input)).detach()
        # latent_rep = np.array(latent_rep, dtype=np.float32)
        classified_as = classifier.predict(latent_rep.reshape(1, -1))[0]
        print('\n##### New Target ####')
        print('CLF PRED {}, Actual {}'.format(classified_as, label))
        if classified_as == 0:
            print('------> Feeding to Low Model')
            y_pred = low_model.predict(input.reshape(1, -1))
            plt.scatter(target, y_pred, s=400, c='salmon', edgecolors=(0,0,0))
        else:
            print('------> Feeding to High Model')
            y_pred = high_model.predict(input.reshape(1, -1))
            plt.scatter(target, y_pred, s=400, c='forestgreen', edgecolors=(0,0,0))

        print('Predicted {}, Actual {}'.format(y_pred, target))


    plt.show()

def main(**kwargs):
    dataset, input_scaler = utils.load_data_torch()
    low_neped, high_neped, final_exam = dataset

    # Train AE
    AE = train_autoencoder(datasets=(low_neped, high_neped, final_exam), model_type=args.vae_type, **kwargs)
    # plot_latent_tsne(datasets=(low_neped, high_neped, final_exam), autoencoder=AE)
    # Train CLF
    CLF = train_classifier(dataset=(low_neped, high_neped, final_exam), autoencoder=AE)

    # Train Low and High Neped models

    LnM = train_regressor(dataset=(low_neped, final_exam))
    HnM = train_regressor(dataset=(high_neped, final_exam))

    # make_pretty_plot(datasets = dataset, autoencoder=AE, classifier=CLF, low_model=LnM, high_model=HnM)
    # Pass information
    take_final_exam(final_exam, AE, CLF, LnM, HnM)
    # take_final_exam(low_neped, AE, CLF, LnM, HnM)
    # take_final_exam(high_neped, AE, CLF, LnM, HnM)




    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Great Split Experiment for VAE models')
    parser.add_argument("-bs", "--batch_size", help='batch size for training', type=int, default=419)
    parser.add_argument("-ep", "--epochs", help='num epochs', type=int, default=172)
    parser.add_argument("-lr","--learning_rate", help='learning rate', type=float, default=0.00039)
    parser.add_argument("-ld", "--latent_dim", help='Latent Dimensions of AE', default=88, type=int)
    parser.add_argument('-hslist', '--hidden_dims', help='List of hidden layers [h1_size, h2_size, ..., ]', nargs='+', default=[150, 300, 400], type=int)
    parser.add_argument('-cond', '--conditional_inputs', help='List of condiitonal inputs to give, for longer list, see README.md', nargs='+', default=['Tepedheight1019(m-3)'], type=str)
    parser.add_argument('-main_eng', '--main_engineering_inputs', help='List of main eng. inputs to give that will be reconstructed, for longer list, see README.md', nargs='+', default=['Ip(MA)', 'B(T)', 'a(m)', 'averagetriangularity',
                     'P_NBI(MW)', 'P_ICRH(MW)','P_TOTPNBIPohmPICRH-Pshi(MW)',
                     'plasmavolume(m3)', 'q95', 'gasflowrateofmainspecies1022(es)'], type=str)
    parser.add_argument('-data', '--data_loc', help='File location of data you wish to feed', type=str, default='/home/adam/data/seperatrix_dataset.csv')
    parser.add_argument('-log', '--log_dir', help='Location of logging', type=str, default='./vae_exps')
    parser.add_argument('-exp_name', '--experiment_name', help='Name of Experiment', type=str, default='STANDALONE')
    parser.add_argument("-seed", "--torch_seed", help='Set manual seed for reproducability', default=42, type=int)
    parser.add_argument("-vae", "--vae_type", help='Which VAE to use, choose VanillaVAE if you do not know', default='VanillaVAE', type=str)
    parser.add_argument("-wd", "--weight_decay", help='Weight decay in optimizer', default=0.0, type=float)

    args = parser.parse_args()
    config = vars(args)
    torch.manual_seed(42)
    try:
        main(**config)
    except Exception as e:
        raise e
