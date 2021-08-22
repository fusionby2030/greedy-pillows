import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from models import BaseVAE


class SimpleAE(BaseVAE):

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List=None, **kwargs)-> None:
        super(SimpleAE, self).__init__()

        self.latent_dim = latent_dim
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        last_dim = input_dim
        # Build the encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(last_dim, h_dim),
                    nn.LeakyReLU())
            )
            last_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Linear(hidden_dims[-1], input_dim)

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input, and returns mean and variance of latent space.
        :param input: (Tensor) Input tensor to the encoder [BS x input_dim]
        :return: (Tensor) List of latent information [mu, log_var]
        """

        z = self.encoder(input)
        z = self.fc_mu(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent distribution values onto the input space.
        :param z: (Tensor) [BS x LD]
        :return: (Tensor) [BS x input_dim]
        """
        x = self.decoder_input(z)
        x = self.decoder(x)
        result = self.final_layer(x)
        return result

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:

        z = self.encode(input)
        return [self.decode(z), input]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]

        loss = F.mse_loss(recons, input)
        return {"loss": loss}

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input x, returns the reconstructed input_space
        :param x: (Tensor) [BS x input_dim]
        :return: (Tensor) [B x input_dim]
        """

        return self.forward(x)[0]

    def get_latent_space(self, x: torch.Tensor, **kwargs)-> torch.Tensor:
        z = self.encode(input)
        return z

    def plot_latent_space(self, data_loader: torch.utils.data.DataLoader, title: str = '', **kwargs)-> torch.Tensor:
        import matplotlib.pyplot as plt
        if kwargs.get('latent_dims'):
            latent_x, latent_y = latent_dims[0], latent_dims[1]
        else:
            latent_x, latent_y = 0, 1
        for i, (x, y) in enumerate(data_loader):
            z = self.encode(x)
            z = z.detach().numpy()
            plt.scatter(z[:, latent_x], z[:, latent_y], c=y, cmap='Spectral', s=400)
        # plt.clim(1.0, 12)
        plt.colorbar()
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.show()
