import torch
import torch.autograd
import torch.distributions
import torch.distributions.kl
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
import functools

class VAE(nn.Module):
    def __init__(self, num_input=100, num_latent=3, convex_factor=0.05, *args, **kwargs):
        super(VAE, self).__init__()
        self.num_input = num_input
        self.num_latent = num_latent
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(num_input, 50),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(50, 25),
            torch.nn.SiLU(),
            torch.nn.Linear(25, 25),
            torch.nn.SiLU(),
            torch.nn.Linear(25, 12),
            torch.nn.SiLU(),
            torch.nn.Linear(12, 2*num_latent)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(num_latent, 12), 
            torch.nn.SiLU(),
            torch.nn.Linear(12, 25),
            torch.nn.SiLU(),
            torch.nn.Linear(25, 25),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(25, 50),
            torch.nn.SiLU(),
            torch.nn.Linear(50, num_input),
        )
        self.N = torch.distributions.kl.MultivariateNormal(torch.Tensor([[0]]),torch.Tensor([[1]]))
        self.Z = torch.distributions.kl.MultivariateNormal(torch.Tensor([[0]]),torch.Tensor([[1]]))
        self.convex_factor = convex_factor
        self.loggable_losses = {
            'loss': self.loss, 
            'kl_div': self.kl_div, 
            'rec_loss': self.rec_loss
            }

    def forward(self, x):
        mu_and_sigma = self.encoder(x)
        self.mu = mu_and_sigma[...,:self.num_latent]  
        self.sigma = torch.exp(mu_and_sigma[...,self.num_latent:]/2)
        self.eps = torch.autograd.Variable(torch.randn_like(self.mu), requires_grad=False)
        self.z = self.mu + self.eps * self.sigma
        self.Z = torch.distributions.kl.MultivariateNormal(self.mu,torch.diag_embed(self.sigma**2))
        self.N = torch.distributions.kl.MultivariateNormal(torch.zeros_like(self.mu),torch.diag_embed(torch.ones_like(self.sigma)))
        
        x_hat = self.decoder(self.z)
        return x_hat

    def to_waveform(self, z):
        return self.decoder(z)

    def sample_waveform(self, z):
        self.eps = torch.autograd.Variable(torch.randn_like(self.mu), requires_grad=False)
        self.z = self.mu + self.eps * self.sigma
        self.Z.loc = self.mu
        self.Z.scale = self.sigma
        x_hat = self.decoder(self.z)

    def to_latent_space(self, x):
        mu_and_sigma = self.encoder(x)
        self.mu = mu_and_sigma[...,:self.num_latent]  
        self.sigma = torch.exp(mu_and_sigma[...,self.num_latent:]/2)
        return self.mu
    
    def kl_div(self, x=None, x_hat=None):
        return torch.mean(torch.distributions.kl.kl_divergence(self.Z, self.N)) / self.num_latent

    def rec_loss(self, x, x_hat):
        return F.mse_loss(x, x_hat) / F.mse_loss(x,torch.zeros_like(x, device=self.device)) # normalize by RMS power of either

    def loss(self, x, x_hat):
        kl_div = self.convex_factor * self.kl_div() 
        rec_loss = (1-self.convex_factor) * self.rec_loss(x,x_hat)
        return kl_div + rec_loss

    def to(self, *args, **kwargs):
        if 'device' in kwargs.keys():
            device = kwargs['device']
        elif len(args) > 0:
            device = args[0]
        self.device = device
        # pass distributions to device
        # mean = torch.tensor(0, device=device)
        # variance = torch.tensor(1, device=device)
        # self.N = torch.distributions.kl.Normal(mean, variance)
        # self.Z = torch.distributions.kl.Normal(mean, variance)

        # then call overridden method
        return super(VAE, self).to(*args, **kwargs) 
        

class AE(nn.Module):
    def __init__(self, num_input=100, num_latent=3, *args, **kwargs):
        super(AE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(num_input, 50),
            torch.nn.BatchNorm1d(50),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(50, 25),
            torch.nn.BatchNorm1d(25),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(25, 12),
            torch.nn.BatchNorm1d(12),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(12, num_latent),
            torch.nn.ReLU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(num_latent, 12), 
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(12, 25),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(25, 50),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(50, num_input),
        )

        self.loggable_losses={
            'loss', self.loss
            }

    def forward(self, x):
        z = self.encoder(x) 
        x_hat = self.decoder(z)
        return x_hat
    
    def to_latent_space(self, x):
        z = self.encoder(x)
        return z

    def to_waveform(self,z):
        x_hat = self.decoder(z)
        return x_hat

    def loss(self, x, x_hat):
        #TODO: make it configurable
        return torch.nn.MSELoss()(x, x_hat)