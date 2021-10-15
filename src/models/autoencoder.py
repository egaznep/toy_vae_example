import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU

class AE(nn.Module):
    def __init__(self, num_latent=3):
        super(AE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, num_latent)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(num_latent, 12), 
            torch.nn.ReLU(),
            torch.nn.Linear(12, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 100)
        )

    def forward(self, x):
        z = self.encoder(x) 
        x_hat = self.decoder(z)
        return x_hat