import torch
from model import VariationalAutoEncoderWithCNN
from loss import VAELoss
from dataset import get_loaaders
import sys

train_loader, test_loader = get_loaaders(batch_size = 128)
vaeLoss = VAELoss()
VAE = VariationalAutoEncoderWithCNN()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAE = VAE.to(device=DEVICE)
