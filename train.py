import torch
from torch.optim import Adam
from torchvision.utils import save_image
from model import get_model
from loss import VAELoss
import matplotlib.pyplot as plt
from dataset import get_loaaders, test_dataset
from PIL import Image
import random
from tqdm import tqdm

train_loader, test_loader = get_loaaders(batch_size = 1024)
vaeLoss = VAELoss()
VAE = get_model()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = VAE.to(device=DEVICE)
lr = 0.001
EPOCHS = 100
generated_images = "./generated"
optimizer = Adam(lr = lr, params= model.parameters())



for epoch in range(EPOCHS):
    running_loss = 0
    model.train()
    loop = tqdm(train_loader)
    for batch, (x, _) in enumerate(loop):
        x = x.to(DEVICE)
        reconstructed, mean, log_var = model(x)
        current_loss = vaeLoss(reconstructed, x, mean = mean, log_var = log_var)
        optimizer.zero_grad()
        current_loss.backward()
        running_loss += current_loss.item()
        optimizer.step()

    print('Testing a random image')
    model.eval()
    with torch.no_grad():
        n = len(test_dataset)
        i = random.choice(list(range(n)))
        img = test_dataset[i][0].unsqueeze(0)
        recon, _, _ = model(img)
        sample = torch.cat((img,recon), dim= 0)
        save_image(sample, f"./sample_result/{epoch}.png", nrow = 2)
    print(f"Epoch: {epoch + 1} / {EPOCHS}, Loss: {running_loss / len(train_loader)}")



