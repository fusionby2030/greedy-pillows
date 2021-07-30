"""
Here we shall write the VAE model, with CVAE capabilities.

Encoder



Decoder
"""

import torch
import torch.nn as nn

from tqdm import tqdm

from sklearn.metrics import mean_squared_error
class AutoEncoder(nn.Module):
    def __init__(self, input_size=10, hidden_size=10):
        super(AutoEncoder, self).__init__()


        self.Encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, hidden_size)
        )

        self.Decoder = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, input_size)
            )

    def encode(self, x):
        x = self.Encoder(x)
        return x

    def decode(self, x):
        x = self.Decoder(x)
        return x

    def forward(self, x):
        x= self.encode(x)
        x = self.decode(x)
        return x


def fit_epoch(model, train_loader, criterion, optimizer, is_cvae=False):
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    for inputs, labels in train_loader:
        # inputs = inputs.to(DEVICE)
        # labels = one_hot(labels,9).to(DEVICE)
        optimizer.zero_grad()
        if is_cvae:
            outputs, mu, logvar = model(inputs,labels)
            loss = vae_loss_fn(inputs, outputs, mu, logvar)
            loss.backward()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        processed_data += inputs.size(0)

    train_loss = running_loss / processed_data
    return train_loss

def eval_epoch(model, val_loader, criterion, is_cvae=False):
    model.eval()
    running_loss = 0.0
    processed_size = 0
    RMSE = 0.0
    inp,out = [],[]
    for inputs, labels in val_loader:
        # inputs = inputs.to(DEVICE)
        # labels = one_hot(labels,9).to(DEVICE)

        with torch.set_grad_enabled(False):
            if is_cvae:
                outputs, mu, logvar = model(inputs,labels)
                loss = vae_loss_fn(inputs.view(-1,28*28), outputs, mu, logvar)
                loss.backward()
            else:
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

def train(train_loader, val_loader, model, epochs, batch_size, is_cvae=False):
    history = []
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} val_loss {v_loss:0.4f} RMSE: {rmse:0.4f}"
    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        opt = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            train_loss = fit_epoch(model, train_loader, criterion, opt, is_cvae)
            print("loss", train_loss)
            val_loss, avg_rmse = eval_epoch(model, val_loader, criterion, is_cvae)
            history.append((train_loss, val_loss))
            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch+1, t_loss=train_loss, v_loss=val_loss, rmse=avg_rmse))
    return history

from data import utils

def main(args):
    dataset, ss = utils.load_data_torch()
    print(dataset)
    split = int(0.7*len(dataset[0]))
    X_train, y_train = dataset[0][split:], dataset[1][split:]
    X_test, y_test = dataset[0][:split], dataset[1][:split]
    train_set = utils.ANNtorchdataset(X_train, y_train)
    test_set = utils.ANNtorchdataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_set)

    model = AutoEncoder()
    crietrion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    epochs = args.epochs
    history = []
    model.double()
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} val_loss {v_loss:0.4f} RMSE: {rmse:0.4f}"
    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        opt = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            train_loss = fit_epoch(model, train_loader, criterion, opt)
            # print("loss", train_loss)
            val_loss, avg_rmse = eval_epoch(model, test_loader, criterion)
            history.append((train_loss, val_loss))
            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch+1, t_loss=train_loss, v_loss=val_loss, rmse=avg_rmse))




import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", help='batch size for training', type=int, default=300)
    parser.add_argument("-ep", "--epochs", help='num epochs', type=int, default=100)
    parser.add_argument("-lr","--learning_rate", help='learning rate', type=float, default=0.01)
    args = parser.parse_args()
    main(args)
