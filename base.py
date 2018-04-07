import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import PIL
from PIL import Image
from torchvision.utils import save_image as save
import random
import visdom
import numpy as np
from mnist_gan_experiment import GAN_Building
from torchvision import datasets

##### Simple classification model

class DataBuilder(data.Dataset):

    def __init__(self, mode="training", split=1.0):

        if mode == "training":
            self.loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,transform=transforms.Compose(
                                                                           [transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])), batch_size=16, shuffle=True)
        else:
            self.loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=16, shuffle=True)

        self.ut = transforms.Compose([transforms.Normalize((-0.1307/0.3081,), (1/0.3081,))])
        self.my_items = []
        pair_state = False
        for i, item_A in enumerate(self.loader):
            if i > split*len(self.loader):
                break

            if not pair_state:
                item_first = item_A
                pair_state = True
            else:
                diff_label = torch.fmod(torch.add(item_first[1], item_A[1]),2)
                self.my_items.append(((item_first[0], item_A[0]),(item_first[1], item_A[1]), (diff_label)))
                pair_state = False

    def __getitem__(self, index):
        datum = self.my_items[index]
        return {'Image Pairs':datum[0], 'Label Pairs':datum[1], 'Diff Labels':datum[2]}

    def __len__(self):
        return len(self.my_items)

    def visualize_pair(self,datum_A, datum_B, label_A, label_B):
        rel = 'None'
        if abs(int(label_A) + int(label_B)) % 2 == 0:
            rel = 'same'
        else:
            rel = 'diff'
        opts = {'caption': "{},{} = {}".format(label_A, label_B, rel)}
        vis.images([torch.clamp(self.ut(datum_A), 0, 1).numpy(), torch.clamp(self.ut(datum_B), 0, 1).numpy()], opts=opts)

######################### Data Building

vis = visdom.Visdom()
train_loader = DataBuilder(mode="training", split=0.01)
test_loader = DataBuilder(mode="testing", split=0.1)
# Pair Reconstructions in VAE-k

V = GAN_Building(model_choice="GAN Zodiac", dbs={'train':train_loader, 'test':test_loader}, result_path="/scratch/Jack/research lab/True_Relations/")
V.train()

#### TO-DO
#### GAN reconstructions not blurry, Leaky_ReLU (No relus in fc sofar), check losses



