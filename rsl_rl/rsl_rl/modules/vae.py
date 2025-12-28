import torch
from torch import nn
from torch.nn import functional as F
from typing import List,Union,Any

from torch import nn
from abc import abstractmethod

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def decode(self, input: torch.Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> torch.Tensor:
        pass



class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 out_channels:int,
                 latent_dim: int,
                 estimate_dim: int,
                 encoder_hidden_dims=[128,64],
                 decoder_hidden_dims=[64,128,48],
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        modules.append(
            nn.Sequential(
                nn.Linear(in_channels,encoder_hidden_dims[0]),
                nn.ELU())
        )
        # Build Encoder
        for i in range(len(encoder_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(encoder_hidden_dims[i],encoder_hidden_dims[i+1]),
                    nn.ELU())
            )

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_vel_est = nn.Linear(encoder_hidden_dims[-1], estimate_dim)


        # Build Decoder
        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(latent_dim+estimate_dim, decoder_hidden_dims[0] ),                    
                nn.ELU()
        ))
        for i in range(len(decoder_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(decoder_hidden_dims[i],decoder_hidden_dims[i+1]),                    
                    nn.ELU()
            ))

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                    nn.Linear(decoder_hidden_dims[-1],out_channels))

    @torch.jit.export
    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.Tensor) Input torch.Tensor to encoder [N x C x H x W]
        :return: (torch.Tensor) List of latent codes
        """
        result = self.encoder(input)
        # result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        vel_est = self.fc_vel_est(result)

        return [mu, log_var,vel_est]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (torch.Tensor) [B x D]
        :return: (torch.Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (torch.Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (torch.Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (torch.Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var,vel_est = self.encode(input)
        z = self.reparameterize(mu, log_var)
        out=torch.cat((z,vel_est),dim=1)
        return  [self.decode(out), input, mu, log_var]

    def loss_function(self,
                      recons,
                      input,
                      mu,
                      log_var,
                      real_vel,
                      est_vel,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = recons
        input = input
        mu = mu
        log_var = log_var

        kld_weight = 0.0
        recons_loss =F.mse_loss(recons, input)

        vel_weight = 1.0
        # print("real_vel",real_vel)
        vel_est_loss = F.mse_loss(real_vel,est_vel)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss + vel_est_loss*vel_weight
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach(),"vel_est":vel_est_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        Sample is not supported for multi-head
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def generate_deterministic(self,x,**kwargs)->torch.Tensor:
        """
        Deterministic forward
        """
        mu, log_var,vel_est = self.encode(x)
        # z = self.reparameterize(mu, log_var)
        z = mu
        out=torch.cat((z,vel_est),dim=1)
        return  self.decode(out)