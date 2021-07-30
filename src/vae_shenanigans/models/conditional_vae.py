import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from models import BaseVAE

class ConditionalVAE(BaseVAE):

    def __init__(self, input_dim: int, cond_dim: int, latent_dim: int, hidden_dims: List = None, **kwargs) -> None:
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [300, 400, 500]

        last_dim = input_dim + cond_dim

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

        self.decoder_input = nn.Linear(latent_dim + cond_dim, hidden_dims[-1])
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Linear(hidden_dims[-1], input_dim)

    def encode(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        if kwargs.get('conditions') is not None:
            y = kwargs['conditions']
            input = torch.cat([input, y], dim=1)
        result = self.encoder(input)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

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
        y = kwargs['conditions']
        x = torch.cat([input, y], dim=1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)

        z = torch.cat([z, y], dim=1)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        should add the condition loss parameterization
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        if kwargs.get('epoch') is not None:
            #print('not using kld \n')
            if kwargs['epoch'] < 50:
                loss = recons_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = kwargs['conditions']
        z = torch.randn(num_samples,
                        self.latent_dim, dtype=torch.float64)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)

        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed input_space
        :param x: (Tensor) [BS x input_dim]
        :return: (Tensor) [B x input_dim]
        """

        return self.forward(x)[0]
