import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.batchnorm(self.conv(x)))
    

class FinalTransposeConvolution(nn.Module):
    def __init__(self, **kwargs):
        super(FinalTransposeConvolution, self).__init__()
        self.transpose1 = nn.ConvTranspose2d(10, 128, kernel_size=(4,4), stride=(2,2), padding=(1, 1))
        self.normalize1 = nn.BatchNorm2d(128)
        self.transpose2 = nn.ConvTranspose2d(128, 32, kernel_size=(2, 2), stride= (2,2), padding=0)
        self.normalize2 = nn.BatchNorm2d(32)
        self.unit_conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.reshape(x.shape[0], 10, 7, 7)
        x = self.activation(self.normalize1(self.transpose1(x)))
        x = self.activation(self.normalize2(self.transpose2(x)))
        return F.sigmoid(self.unit_conv(x))


class VariationalAutoEncoderWithCNN(nn.Module):
    def __init__(self, latent_size = 40, input_size =(1, 28,28)):
        super(VariationalAutoEncoderWithCNN, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.encode_block = self._create_encode_layers()
        self.fc_mean = nn.Linear(latent_size, latent_size)
        self.fc_log_var = nn.Linear(latent_size, latent_size)
        self.decode_block = self._create_decode_block()


    def _create_encode_layers(self):
        layers = [
            CNNBlock(in_channels=1, out_channels=32, stride = 1, padding = 1, kernel_size = 3),
            nn.MaxPool2d(kernel_size=(2, 2), stride = (2, 2)),
            CNNBlock(in_channels=32, out_channels=128, stride = 1, padding = 1, kernel_size = 3),
            nn.MaxPool2d(kernel_size=(2, 2), stride = (2, 2)),
            nn.AvgPool2d(7),
            nn.Flatten(start_dim=1),
            nn.Linear(128, self.latent_size)
        ]
        return nn.Sequential(*layers)

    def fn_reparameterize(self, mean, log_var):
        std = torch.exp(log_var/ 2)
        eps = torch.randn_like(std)
        return (std * eps) + mean

    def _create_decode_block(self):
        transposed_conv = FinalTransposeConvolution()
        layers = [
            nn.Linear(self.latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 490),
            nn.ReLU(),
            transposed_conv,
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        out  = self.encode_block(x)
        mean = self.fc_mean(out)
        log_var = self.fc_log_var(out)
        latent = self.fn_reparameterize(mean = mean, log_var=log_var)
        reconstructed = self.decode_block(latent)
        return reconstructed, mean, log_var


