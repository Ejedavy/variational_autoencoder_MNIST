import torch
import torch.nn as nn


"""
This contains the loss calculations for the construction i.e Binary cross entropy and the KL Divergence loss to ensure that the encoder
learns a wider variance after the reparametrization trick.
"""

class VAELoss(nn.Module):
    def __init__(self, *args , **kwargs):
        super(VAELoss, self).__init__()
        self.BCELoss = nn.BCELoss(reduction='sum')

    def KLDLoss(self, **kwargs):
        mean = kwargs['mean']
        log_var = kwargs['log_var']
        return 0.5 * torch.sum(log_var.exp()  + mean.pow(2) - 1 - log_var)
    

    def forward(self, prediction , target, **kwargs):
        reconstruction_loss = self.BCELoss(prediction.view(-1, 784), target.view(-1, 784))
        klLoss = self.KLDLoss(**kwargs)
        return reconstruction_loss + klLoss