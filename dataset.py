from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

train_dataset = MNIST(root = "./archive", 
                      train = True, 
                      transform=transforms.ToTensor(),
                      download= True
                      )

test_dataset = MNIST(root = "./archive", 
                     train = False, 
                     transform= transforms.ToTensor(), 
                     download = False)



def get_loaaders(*args, **kwargs):
    train_loader = DataLoader(train_dataset, *args, **kwargs)
    test_loader = DataLoader(test_dataset, *args, **kwargs)
    return train_loader, test_loader

