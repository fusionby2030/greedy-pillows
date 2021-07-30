import argparse
import numpy as np

import torch
from torch import optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch import Tensor
from models import BaseVAE
from typing import List, Callable, Union, Any, TypeVar, Tuple
import warnings
from data import utils


import matplotlib.pyplot as plt
def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        try: # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except: # Works for version > 0.6.0
            return fn(self)

    return func_wrapper

class VAEXperiment(pl.LightningModule):

    def __init__(self, vae_model: BaseVAE, params:dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params


    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)


    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        inputs, targets = batch
        # self.curr_device = real_img.device

        results = self.forward(inputs, conditions=targets)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['batch_size']/ self.num_train_shots,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        inputs, targets = batch

        results = self.forward(inputs, conditions=targets)
        val_loss = self.model.loss_function(*results,
                                              M_N = self.params['batch_size']/ self.num_val_shots,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        # self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        # self.sample_images()
        self.logger.experiment.log_graph(self.model)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['learning_rate'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)

        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims


    @data_loader
    def train_dataloader(self):
        try:
            dataset, ss = utils.load_data_torch(conditions=self.params['conditional_inputs'], main_engineering_inputs=self.params['main_engineering_inputs'],data_loc=self.params['data_loc'])
        except KeyError as e:
            warnings.warn("You did not pass conditions, main engnieering inputs, or data location in the parameters, so using the default")
            dataset, ss = utils.load_data_torch(conditions=self.params['conditional_inputs'], main_engineering_inputs=self.params['main_engineering_inputs'],data_loc=self.params['data_loc'])
            self.input_scaler = ss
        if self.params.get('split') is not None:
            X_train, y_train = dataset[0][:split], dataset[1][:split]
        else:
            split = int(0.7*len(dataset[0]))
            X_train, y_train = dataset[0][:split], dataset[1][:split]
        train_set = utils.ANNtorchdataset(X_train, y_train)
        self.num_train_shots = len(dataset[0])

        return DataLoader(train_set, self.params['batch_size'], shuffle=True)

    @data_loader
    def val_dataloader(self):

        try:
            dataset, ss = utils.load_data_torch(conditions=self.params['conditional_inputs'], main_engineering_inputs=self.params['main_engineering_inputs'],data_loc=self.params['data_loc'])
        except KeyError as e:
            warnings.warn("You did not pass conditions, main engnieering inputs, or data location in the parameters, so using the default")
            dataset, ss = utils.load_data_torch(conditions=self.params['conditional_inputs'], main_engineering_inputs=self.params['main_engineering_inputs'],data_loc=self.params['data_loc'])
            self.input_scaler = ss
        if self.params.get('split') is not None:
            X_train, y_train = dataset[0][split:], dataset[1][split:]
        else:
            split = int(0.7*len(dataset[0]))
            X_train, y_train = dataset[0][split:], dataset[1][split:]
        train_set = utils.ANNtorchdataset(X_train, y_train)
        self.num_val_shots = len(X_train)

        return DataLoader(train_set, self.params['batch_size'], shuffle=False)
