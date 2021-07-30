"""
A simple implenetation of autoencoder for JET Pedestal database.
"""
# Data
from data import utils

# Torch stuffs
import torch
import torch.nn as nn

# Misc
import argparse
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
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

"""
First we define the Encoder and Decoder classes.

Encoder - 2 layers
"""

class Encoder(nn.Module):
    def __init__(self, input_dims=10, latent_dims=20):
        super(Encoder, self).__init__()

        self.block = nn.Sequential(
        nn.Linear(input_dims, 300),
        nn.ELU(),
        nn.Linear(300, 100),
        nn.ELU(),
        nn.Linear(100, latent_dims))

    def forward(self, x):
        z = self.block(x)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims=2, output_dims=10):
        super(Decoder, self).__init__()
        self.block = nn.Sequential(
        nn.Linear(latent_dims, 300), nn.ELU(),
        nn.Linear(300, 100), nn.ELU(),
        nn.Linear(100, output_dims))

    def forward(self, z):
        x = self.block(z)
        return x


"""
Autoencoder Class
"""

class Autoencoder(nn.Module):
    def __init__(self, latent_dims=2, input_dims=10):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dims, latent_dims)
        self.decoder = Decoder(latent_dims, input_dims)

    def forward(self, x):
        z = self.encoder(x)
        x_hat= self.decoder(z)
        return x_hat


"""
Training Function
"""

def eval_epoch(model, val_loader, criterion, is_cvae=False):
    model.eval()
    running_loss = 0.0
    processed_size = 0
    RMSE = 0.0
    inp,out = [],[]
    for inputs, labels in val_loader:
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            inp,out = inputs, outputs
            rmse = mean_squared_error(y_true=inputs, y_pred=outputs, squared=False)
        RMSE += rmse*inputs.size(0)
        running_loss += loss.item() * inputs.size(0)
        processed_size += inputs.size(0)

    val_loss = running_loss / processed_size
    avg_RMSE = RMSE / processed_size
    return val_loss, avg_RMSE

def train(autoencoder, train_loader, test_loader=None, optimizer=None, epochs=100, criterion=None):
    if optimizer == None:
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
    if criterion == None:
        criterion = torch.nn.MSELoss()
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} val_loss {v_loss:0.4f} RMSE: {rmse:0.4f}"
    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for epoch in range(epochs):
            for batch_idx, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                x_hat = autoencoder(x)
                loss = criterion(x, x_hat)
                loss.backward()
                optimizer.step()
            if test_loader:
                val_loss, avg_RMSE = eval_epoch(autoencoder, test_loader, criterion)
            else:
                val_loss = avg_RMSE = 0.0
            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch+1, t_loss=loss.item(), v_loss=val_loss, rmse=avg_RMSE))

    return autoencoder

def plot_latent(autoencoder, data_loader, title=None):
    fig = plt.figure(figsize=(18, 18))
    for i, (x, y) in enumerate(data_loader):
        z = autoencoder.encoder(x)
        z = z.detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='Spectral', s=400)
    # plt.clim(1.0, 12)
    plt.colorbar()
    if title:
        plt.title(title)
    plt.tight_layout()

def plot_train_and_test(autoencoder, train_loader, test_loader, title=None):

    plt.figure(figsize=(36, 18))

    # fig, axs = plt.subplots(1, 2, figsize=(18, 18), sharex=True, sharey=True)
    plt.subplot(2, 2, 1)
    x_vals = y_vals = []
    for i, (x, y) in enumerate(train_loader):
        z = autoencoder.encoder(x)
        z = z.detach().numpy()
        if y > 10.5:
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='Spectral', s=400, vmin=1.8494685, vmax=11.737379, zorder=3)
        elif y > 9.5:
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='Spectral', s=400, vmin=1.8494685, vmax=11.737379, zorder=2)
        else:
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='Spectral', s=400, vmin=1.8494685, vmax=11.737379, zorder=1)

    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.title('Train Set')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    for i, (x, y) in enumerate(test_loader):
        z = autoencoder.encoder(x)
        z = z.detach().numpy()
        if y > 10.5:
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='Spectral', s=400, vmin=1.8494685, vmax=11.737379, zorder=3)
        elif y > 9.5:
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='Spectral', s=400, vmin=1.8494685, vmax=11.737379, zorder=2)
        else:
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='Spectral', s=400, vmin=1.8494685, vmax=11.737379, zorder=1)


    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.title('Test Set')


    plt.subplot(2, 2, 3)
    for i, (x, y) in enumerate(train_loader):
        z = autoencoder.encoder(x)
        z = z.detach().numpy()
        if y > 10.5:
            plt.scatter(z[:, 1], z[:, 2], c=y, cmap='Spectral', s=400, vmin=1.8494685, vmax=11.737379, zorder=3)
        elif y > 9.5:
            plt.scatter(z[:, 1], z[:, 2], c=y, cmap='Spectral', s=400, vmin=1.8494685, vmax=11.737379, zorder=2)
        else:
            plt.scatter(z[:, 1], z[:, 2], c=y, cmap='Spectral', s=400, vmin=1.8494685, vmax=11.737379, zorder=1)

    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.subplot(2, 2, 4)
    for i, (x, y) in enumerate(test_loader):
        z = autoencoder.encoder(x)
        z = z.detach().numpy()
        if y > 10.5:
            plt.scatter(z[:, 1], z[:, 2], c=y, cmap='Spectral', s=400, vmin=1.8494685, vmax=11.737379, zorder=3)
        elif y > 9.5:
            plt.scatter(z[:, 1], z[:, 2], c=y, cmap='Spectral', s=400, vmin=1.8494685, vmax=11.737379, zorder=2)
        else:
            plt.scatter(z[:, 1], z[:, 2], c=y, cmap='Spectral', s=400, vmin=1.8494685, vmax=11.737379, zorder=1)

    plt.ylim(-10, 10)
    plt.xlim(-10, 10)

    plt.colorbar(extend='both')
    plt.tight_layout()


def main(args):
    # Get Data Loaders
    dataset, ss = utils.load_data_torch()
    split = int(1*len(dataset[0]))
    X_train, y_train = dataset[0][:split], dataset[1][:split]
    X_test, y_test = dataset[0][split:], dataset[1][split:]
    print(X_train.shape)
    print(X_test.shape)
    train_set = utils.ANNtorchdataset(X_train, y_train)
    test_set = utils.ANNtorchdataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, args.batch_size)

    # Setup Autoencoder
    autoencoder = Autoencoder(latent_dims=args.latent_dims)
    autoencoder.double()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.learning_rate)
    autoencoder = train(autoencoder, train_loader, test_loader=None, optimizer=optimizer, epochs=args.epochs)

    plot_latent(autoencoder, train_loader, title="AE - 2 Latent Dims")
    # plot_latent(autoencoder, test_loader, title="Test Set")
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=1)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
    # plot_train_and_test(autoencoder, train_loader, test_loader)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Shenanigans')

    parser.add_argument("-bs", "--batch_size", help='batch size for training', type=int, default=300)
    parser.add_argument("-ep", "--epochs", help='num epochs', type=int, default=50)
    parser.add_argument("-lr","--learning_rate", help='learning rate', type=float, default=0.01)
    parser.add_argument("-ld", "--latent_dims", help='Latent Dimensions of AE', default=2, type=int)

    args = parser.parse_args()
    torch.manual_seed(42)
    main(args)
