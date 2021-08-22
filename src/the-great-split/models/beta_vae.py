import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple


class BetaVAE(BaseVAE):

    num_iter = 0

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List = None, beta: int=4, gamma: float= 1000., max_capacity: int=25, Capacity_max_iter: int=1e5, loss_type:str = 'B', **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build the encoder
        last_dim = input_dim

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(last_dim, h_dim),
                    nn.LeakyReLU())
            )
            last_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
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

        """
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], input_dim),
            nn.ReLU())
        """
        self.final_layer = nn.Linear(hidden_dims[-1], input_dim)

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input, and returns mean and variance of latent space.
        :param input: (Tensor) Input tensor to the encoder [BS x input_dim]
        :return: (Tensor) List of latent information [mu, log_var]
        """

        z = self.encoder(input)

        mu = self.fc_mu(z)
        log_var = self.fc_var(z)

        return [mu, log_var]

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

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [BS x LD]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [BS x LD]
        :return: (Tensor) [BS x LD]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim, dtype=torch.float64)


        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed input_space
        :param x: (Tensor) [BS x input_dim]
        :return: (Tensor) [B x input_dim]
        """

        return self.forward(x)[0]
