import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import PIL
from PIL import Image
from torchvision.utils import save_image as save
import random
from models import *

from torchvision import datasets

######################### Data Building

train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])), batch_size=32, shuffle=True)

for item in train_loader():
    print(item)