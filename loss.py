import torch
import torch.nn as nn


"""
This contains the loss calculations for the construction i.e Binary cross entropy and the KL Divergence loss to ensure that the encoder
learns a wider variance after the reparametrization trick.
"""

class VAELoss(nn.Module):
    def __init(self, *args , **kwargs):
        super(VAELoss, self).__init__()
        self.BCELoss = nn.BCELoss()
        self.KLDivergence = self.KLDLoss()

    def KLDLoss(self):
        return
    

    def forward(self, prediction , target, **kwargs):
        reconstruction_loss = self.BCELoss(prediction, target)
        klLoss = self.KLDivergence(prediction, target, **kwargs)
        return reconstruction_loss + klLoss