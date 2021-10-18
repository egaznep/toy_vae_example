import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU

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