import torch
import torch.nn as nn
import random
from torch.autograd import Variable
import math
from sklearn.cluster import KMeans
import pathlib
from sklearn.cluster import KMeans
import progressbar
import torch.optim as optim
import math
import random
import visdom
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
import os.path as path
import PIL
from PIL import Image
import torch.backends.cudnn as tbc
import glob

tbc.benchmark= True

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class decoder(nn.Module):
    # initializers
    def __init__(self, image_size=64, density=128, latent_dim=100, channels=3):
        super(decoder, self).__init__()

        depth = math.log2(image_size) # image size is 64
        assert depth == round(depth)
        assert depth >= 3
        depth = int(depth)

        self.decoder = nn.Sequential()
        self.decoder.add_module("input convolution(t)", nn.ConvTranspose2d(latent_dim, int(density*2**(depth-3)), 4, 1, 0, bias=False))
        self.decoder.add_module("input batchnorm", nn.BatchNorm2d(int(density*2**(depth-3))))
        self.decoder.add_module("input relu", nn.LeakyReLU(0.2, inplace=True))

        for layer in range(depth-3, 0, -1):
            self.decoder.add_module('decoder conv {0}-{1}'.format(density * 2 ** layer ,density * 2 ** (layer - 1)), nn.ConvTranspose2d(int(density * 2 ** (layer)), int(density * 2 ** (layer - 1)), 4, 2, 1, bias=False))
            self.decoder.add_module('decoder batchnorm {0}'.format(density * 2 ** (layer - 1)), nn.BatchNorm2d(int(density * 2 ** (layer - 1))))
            self.decoder.add_module('decoder relu {0}'.format(density * 2 ** (layer - 1)), nn.LeakyReLU(0.2, inplace=True))

        self.decoder.add_module('ouput conv', nn.ConvTranspose2d(density, channels, 4, 2, 1, bias=False))
        self.decoder.add_module('output tanh', nn.Tanh())

    # forward method
    def forward(self, input):
        output = self.decoder(input)
        return output

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class encoder(nn.Module):
    # initializers
    def __init__(self, image_size=64, density=128, latent_dim=99, latent_group = 33,channels=3):
        super(encoder, self).__init__()

        depth = math.log2(image_size) # image size is 64
        assert depth == round(depth)
        assert depth >= 3
        depth = int(depth)

        self.latent_dim = latent_dim
        self.latent_group = latent_group
        self.clusters = int(self.latent_dim // self.latent_group)

        self.conv_mean = nn.Conv2d(int(density*2**(depth-3)), latent_dim, 4)
        self.conv_logvar = nn.Conv2d(int(density*2**(depth-3)), latent_dim, 4)
        self.encoder = nn.Sequential()
        self.encoder.add_module("input convolution", nn.Conv2d(channels, density, 4, 2, 1, bias=False))
        self.encoder.add_module("input relu", nn.LeakyReLU(0.2, inplace=True))

        for layer in range(depth-3):
            self.encoder.add_module('encoder conv {0}-{1}'.format(density * 2 ** layer ,density * 2 ** (layer + 1)), nn.Conv2d(int(density * 2 ** (layer)), int(density * 2 ** (layer + 1)), 4, 2, 1, bias=False))
            self.encoder.add_module('encoder batchnorm {0}'.format(density * 2 ** (layer + 1)), nn.BatchNorm2d(int(density * 2 ** (layer + 1))))
            self.encoder.add_module('encoder relu {0}'.format(density * 2 ** (layer + 1)), nn.LeakyReLU(0.2, inplace=True))

    # forward method
    def forward(self, input):
        output = self.encoder(input)

        mean = self.conv_mean(output)
        logvar = self.conv_logvar(output)

        mu_list = [mean[:,0+i*self.latent_group:self.latent_group+i*self.latent_group] for i in range(self.clusters)]
        logvar_list = [logvar[:,0+i*self.latent_group:self.latent_group+i*self.latent_group] for i in range(self.clusters)]
        return [mu_list, logvar_list]

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class discriminator(nn.Module):
    # initializers
    def __init__(self, image_size=64, density=128, latent_dim=99, channels=3):
        super(discriminator, self).__init__()

        depth = math.log2(image_size) # image size is 64
        assert depth == round(depth)
        assert depth >= 3
        depth = int(depth)

        self.discriminator = nn.Sequential()
        self.discriminator.add_module("input convolution", nn.Conv2d(channels, density, 4, 2, 1, bias=False))
        self.discriminator.add_module("input relu", nn.LeakyReLU(0.2, inplace=True))

        for layer in range(depth-3):
            self.discriminator.add_module('encoder conv {0}-{1}'.format(density * 2 ** layer ,density * 2 ** (layer + 1)), nn.Conv2d(int(density * 2 ** (layer)), int(density * 2 ** (layer + 1)), 4, 2, 1, bias=False))
            self.discriminator.add_module('encoder batchnorm {0}'.format(density * 2 ** (layer + 1)), nn.BatchNorm2d(int(density * 2 ** (layer + 1))))
            self.discriminator.add_module('encoder relu {0}'.format(density * 2 ** (layer + 1)), nn.LeakyReLU(0.2, inplace=True))


        self.discriminator.add_module('output-conv', nn.Conv2d(density * 2**(depth-3), 1, 4, 1, 0, bias=False))
        self.discriminator.add_module('output-sigmoid', nn.Sigmoid())

    # forward method
    def forward(self, input):

        output = self.discriminator(input)
        return output.view(-1, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class RelationPrediction(nn.Module):
    def __init__(self, z, c, h, streams, classes=16, domains=5):
        super(RelationPrediction, self).__init__()
        self.h = h
        self.z = z
        self.c = c
        self.streams = streams
        self.classes = classes
        self.domains = domains
        self.full_FC = nn.Sequential(nn.Linear(self.z*self.streams, self.h), nn.Linear(self.h, self.h), nn.Linear(self.h, self.classes))
        self.full_FC_domain = nn.Sequential(nn.Linear(self.z*self.streams, self.h), nn.Linear(self.h, self.h), nn.Linear(self.h, self.domains))

    def forward(self, datum):
        z_vectors = []
        for s in range(self.streams):
            z_vectors.append(datum[s])
        z_vector_cat = torch.cat(z_vectors,1).squeeze()
        class_distribution = self.full_FC(z_vector_cat)
        class_distribution_domain = self.full_FC_domain(z_vector_cat)
        return class_distribution, class_distribution_domain

class FaceDiscriminator(nn.Module):
    def __init__(self, complexity="Complexity A", clusters=3, streams=4, channels=3, image_size=64):

        super(FaceDiscriminator, self).__init__()
        self.complexity = complexity
        self.clusters = clusters
        self.streams = streams
        self.channels = channels

        if self.complexity == "Complexity A":
            self.discriminate = discriminator(image_size=image_size)

    def forward(self, discriminate_datums):

        discriminator_predictions = [None for i in range(len(discriminate_datums))]
        for stream in range(len(discriminate_datums)):
            discriminator_predictions[stream] = self.discriminate(discriminate_datums[stream])

        return discriminator_predictions

class RecombinedGAN(nn.Module):

    def __init__(self, complexity="Complexity A", proper_size=(64,64), channels = 3, hidden=512):
        super(RecombinedGAN, self).__init__()
        self.complexity = complexity
        self.proper_size = proper_size
        self.channels = channels
        self.hidden = hidden
        self.GAN = nn.ModuleList([FaceVAEMixture(complexity=self.complexity, proper_size=self.proper_size, channels=self.channels), FaceDiscriminator(complexity=self.complexity, channels=self.channels, image_size=proper_size[0])])

    def forward(self, datums):
        assert "Use partial modules instead..." == "You are not using partial modules..."
        return 14

class FaceVAEMixture(nn.Module):

    def sample_z(self, input):

        mu = input[0]
        logvar = input[1]

        std = logvar.mul(0.5).exp_()
        eps  = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        # Using reparameterization trick to sample from a gaussian mixture
        if self.inference:
            return mu

        return eps.mul(std).add_(mu)

    def __init__(self, complexity="Complexity A", clusters=3, streams = 4, proper_size=(64,64), channels=3):
        super(FaceVAEMixture, self).__init__()
        self.inference = False
        self.complexity = complexity
        self.clusters = clusters
        self.streams = streams
        self.channels = channels
        self.proper_size = proper_size
        self.log_string = None

        if self.complexity == "Complexity A":
            self.z = 99
            assert self.z % self.clusters == 0
            self.hidden = 128
            self.encoder = encoder(image_size=self.proper_size[0], density=64, latent_dim=self.z, latent_group = int(self.z//self.clusters), channels=self.channels)
            self.decoder = decoder(image_size=self.proper_size[0], density=64, latent_dim=self.z, channels=self.channels)
            self.classify = RelationPrediction(self.z, self.clusters, self.hidden, self.streams//2, classes=16, domains=5)

        #self.upsample = nn.Upsample(scale_factor=8)

    def log(self, s):
        if self.log_string is None:
            self.log_string = '\n'
        self.log_string += s
        self.log_string += '\n'

    def terminate(self):
        print(self.log_string)
        exit()

    def forward(self, datums):

        stream_inputs = [stream for stream in datums]
        stream_h_cat = [None for i in range(len(datums))]
        stream_outputs = [None for i in range(len(datums))]
        stream_outputs_tanh = [None for i in range(len(datums))]

        mu_lists = [[] for i in range(len(datums))]
        std_lists = [[] for i in range(len(datums))]
        z_sample_lists = [[] for i in range(len(datums))]

        for stream_idx in range(len(datums)):
            mu_list, std_list = self.encoder(stream_inputs[stream_idx])

            mu_lists[stream_idx] = mu_list # list of clusters containing mu vectors
            std_lists[stream_idx] = std_list
            z_sample_list = []

            for i in range(self.clusters):
                z_sample_list.append(self.sample_z([mu_list[i], std_list[i]]))
            z_sample_lists[stream_idx] = z_sample_list

            # Each z sample for a particular stream is a list of vectors of size B x group z dimension x 1 x 1

            # z_sample_list are the separate samples of z for the different clustering VAEs
            # combine them together, to form stream_h_cat which reconstructs a single image
            stream_h_cat[stream_idx] = torch.cat(z_sample_lists[stream_idx],1)

            stream_outputs[stream_idx] = self.decoder(stream_h_cat[stream_idx])

        # Create a recombined sample for each pair in the batch (assume pair of 2 streams)
        z_pair_A = z_sample_lists[0]
        z_pair_B = z_sample_lists[1]
        z_pair_random_A = z_sample_lists[2]
        z_pair_random_B = z_sample_lists[3]

        random_cluster = random.randint(0,self.clusters)
        z_Recombined = [Variable(torch.zeros(self.z//self.clusters, 1, 1)).cuda() for k in range(self.clusters)]

        for c in range(self.clusters):
            if random_cluster == c:
                z_Recombined[c] = torch.add(z_Recombined[c], z_pair_A[c])
            else:
                z_Recombined[c] = torch.add(z_Recombined[c], z_pair_random_A[c])

        rec = [] # Recombined list of streams of pairs

        r = torch.cat(z_Recombined, 1)
        rec.append(self.decoder(r))

        random_cluster = random.randint(0, self.clusters)
        z_Recombined2 = [Variable(torch.zeros(self.z//self.clusters, 1, 1)).cuda() for k in range(self.clusters)]
        for c in range(self.clusters):
            if random_cluster == c:
                z_Recombined2[c] = torch.add(z_Recombined2[c], z_pair_B[c])
            else:
                z_Recombined2[c] = torch.add(z_Recombined2[c], z_pair_random_B[c])

        r2 = torch.cat(z_Recombined2, 1)
        rec.append(self.decoder(r2))

        class_prediction, domain_prediction = self.classify([stream_h_cat[0], stream_h_cat[1]]) # Classify the relationship based on the real faces

        return stream_outputs, stream_inputs, rec, mu_lists, std_lists, z_sample_lists, class_prediction, domain_prediction


    def decode(self, z_sample_lists):
        # input: a list of latent z's for each cluster
        stream_h_cat = torch.cat(z_sample_lists, 1)
        stream_out = self.decoder(stream_h_cat)
        stream_recons = stream_out
        stream_ts = self.tanh_out(stream_out)
        stream_outputs = stream_ts
        return stream_outputs, stream_recons, z_sample_lists

