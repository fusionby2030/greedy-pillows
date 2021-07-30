from data import utils

import torch
import torch.nn as nn

from models.vanilla_vae import VanillaVAE
from models.conditional_vae import ConditionalVAE

from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
import argparse
import numpy as np

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


def plot_tsne(model, data_set, title='VAE, TSNE'):
    fig = plt.figure(figsize=(18, 18))
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=len(data_set))
    for batch_idx, (x,y) in enumerate(data_loader):
        latent_codes = model.encode(x, conditions=y)
        z = model.reparameterize(*latent_codes)
        z = z.detach().numpy()
        z = TSNE(n_components=2,random_state=42).fit_transform(z)
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='Spectral', s=400)

    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.colorbar()
    plt.title(title)
    plt.show()

def plot_latent_space(model, data_set, title='VAE, 2 Latent Dimensions'):
    fig = plt.figure(figsize=(18, 18))
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=len(data_set))
    for batch_idx, (x,y) in enumerate(data_loader):
        latent_codes = model.encode(x, conditions=y)
        z = model.reparameterize(*latent_codes)
        z = z.detach().numpy()
        print(z)
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='Spectral', s=400)

    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.colorbar()
    plt.title(title)
    plt.show()

def main_cvae(args):
    """ Need to rework the utils such that we can add conditions outside of neped"""

    dataset, ss = utils.load_data_torch(conditions=args.conditional_inputs, main_engineering_inputs=args.main_engineering_inputs, data_loc = args.data_loc)
    split = int(1.0*len(dataset[0]))
    X_train, y_train = dataset[0][:split], dataset[1][:split]
    X_test, y_test = dataset[0][split:], dataset[1][split:]
    train_set = utils.ANNtorchdataset(X_train, y_train)
    test_set = utils.ANNtorchdataset(X_test, y_test) if split != len(dataset[0]) else None
    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, args.batch_size, shuffle=True) if test_set is not None else None

    cvae = ConditionalVAE(input_dim=10, cond_dim=1, latent_dim=args.latent_dim, hidden_dims=args.hidden_layer_sizes)
    cvae.double()

    optimizer = torch.optim.Adam(cvae.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        cvae.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            results = cvae.forward(x, conditions=y)
            train_loss = cvae.loss_function(*results, M_N = args.batch_size)
            train_loss['loss'].backward()
            optimizer.step()
            # print(train_loss)
            #print('\n')
            print(train_loss['loss'].item(), train_loss['Reconstruction_Loss'].item(), train_loss['KLD'].item())
        print(f'Epoch {epoch}\n')
        if test_loader is not None:
            vae.eval()
            print('Valid \n')
            for batch_idx, (x, y) in enumerate(test_loader):
                results = cvae.forward(x, conditions=y)
                val_loss = cvae.loss_function(*results, kwargs= {'M_N':args.batch_size, 'epoch':epoch})
                # print(val_loss)
                #print('\n')
                print(val_loss['loss'].item(), val_loss['Reconstruction_Loss'].item(), val_loss['KLD'].item())
                print(mean_squared_error(y_true=results[1].detach().numpy(), y_pred=results[0].detach().numpy(), squared=False))
            print('Train \n')
    cvae.double()

    #print(torch.from_numpy(np.linspace(2, 12, num=100)))
    #print(torch.from_numpy(np.linspace(2, 12, num=100)).unsqueeze(1))

    samples = cvae.sample(100, conditions=torch.from_numpy(np.linspace(2, 12, num=100)).unsqueeze(1))
    np_samples = samples.detach().numpy()

    # print(np.mean(X_train, axis=0))
    print(ss.inverse_transform(np.mean(X_train, axis=0)))
    print(np.std(ss.inverse_transform(X_train), axis=0))

    print(np.mean(ss.inverse_transform(np_samples), axis=0))
    print(np.std(ss.inverse_transform(np_samples), axis=0))

    plot_tsne(cvae,train_set)
    plot_latent_space(cvae, train_set)


def main(args):
    dataset, ss = utils.load_data_torch()
    split = int(1.0*len(dataset[0]))
    X_train, y_train = dataset[0][:split], dataset[1][:split]
    X_test, y_test = dataset[0][split:], dataset[1][split:]
    train_set = utils.ANNtorchdataset(X_train, y_train)
    test_set = utils.ANNtorchdataset(X_test, y_test) if split != len(dataset[0]) else None
    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, args.batch_size, shuffle=True) if test_set is not None else None

    vae = VanillaVAE(input_dim=10, latent_dim=args.latent_dim, hidden_dims=args.hidden_layer_sizes)
    vae.double()
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        vae.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            results = vae.forward(x)
            train_loss = vae.loss_function(*results, M_N = args.batch_size, epoch=epoch)
            train_loss['loss'].backward()
            optimizer.step()
            # print(train_loss)
            #print('\n')
            print(train_loss['loss'].item(), train_loss['Reconstruction_Loss'].item(), train_loss['KLD'].item())
        print(f'Epoch {epoch}\n')
        if test_loader is not None:
            vae.eval()
            print('Valid \n')
            for batch_idx, (x, y) in enumerate(test_loader):
                results = vae.forward(x)
                val_loss = vae.loss_function(*results, kwargs= {'M_N':args.batch_size, 'epoch':epoch})
                # print(val_loss)
                #print('\n')
                print(val_loss['loss'].item(), val_loss['Reconstruction_Loss'].item(), val_loss['KLD'].item())
                print(mean_squared_error(y_true=results[1].detach().numpy(), y_pred=results[0].detach().numpy(), squared=False))
            print('Train \n')
    vae.double()
    samples = vae.sample(100)
    np_samples = samples.detach().numpy()

    plot_latent_space(vae, train_set)

    # print(np.mean(X_train, axis=0))
    print(ss.inverse_transform(np.mean(X_train, axis=0)))
    print(np.std(ss.inverse_transform(X_train), axis=0))

    print(np.mean(ss.inverse_transform(np_samples), axis=0))
    print(np.std(ss.inverse_transform(np_samples), axis=0))

    """
    print(samples[0])
    print(ss.inverse_transform(samples[0].detach().numpy()))
    print(samples[1])
    print(ss.inverse_transform(samples[1].detach().numpy()))
    print(samples[2])
    print(ss.inverse_transform(samples[2].detach().numpy()))
    print(torch.mean(samples, axis=0), torch.std(samples, axis=0))"""




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Shenanigans')

    parser.add_argument("-bs", "--batch_size", help='batch size for training', type=int, default=300)
    parser.add_argument("-ep", "--epochs", help='num epochs', type=int, default=50)
    parser.add_argument("-lr","--learning_rate", help='learning rate', type=float, default=0.01)
    parser.add_argument("-ld", "--latent_dim", help='Latent Dimensions of AE', default=2, type=int)
    parser.add_argument('-hslist', '--hidden_layer_sizes', help='List of hidden layers [h1_size, h2_size, ..., ]', nargs='+', default=[150, 300], type=int)
    parser.add_argument('-cond', '--conditional_inputs', help='List of condiitonal inputs to give, for longer list, see README.md', nargs='+', default=['nepedheight1019(m-3)'], type=str)
    parser.add_argument('-main_eng', '--main_engineering_inputs', help='List of main eng. inputs to give that will be reconstructed, for longer list, see README.md', nargs='+', default=['Ip(MA)', 'B(T)', 'a(m)', 'averagetriangularity',
    				 'P_NBI(MW)', 'P_ICRH(MW)','P_TOTPNBIPohmPICRH-Pshi(MW)',
    				 'plasmavolume(m3)', 'q95', 'gasflowrateofmainspecies1022(es)'], type=str)
    parser.add_argument('-data', '--data_loc', help='File location of data you wish to feed', type=str, default='/home/adam/data/seperatrix_dataset.csv')
    args = parser.parse_args()
    torch.manual_seed(42)
    main_cvae(args)
