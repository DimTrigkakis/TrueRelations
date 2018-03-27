import torch
import torch.nn as nn
import random
from torch.autograd import Variable
import math
from resnet_models import BasicBlock, ReverseBasicBlock

class ResnetFragmentDecoder(nn.Module):

    def __init__(self, inplanes, block, layers):
        super(ResnetFragmentDecoder, self).__init__()

        self.original_inplanes = inplanes
        self.inplanes = inplanes
        self.block = block
        self.layers = layers

        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample7 = nn.Upsample(scale_factor=7)
        self.relu = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.ConvTranspose2d(64, 1, kernel_size=7, stride=2, output_padding=1, padding=3,bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1): # ALL BATCH NORMS FROM SECOND PLACE NOT FIRST in prev conv
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(nn.ConvTranspose2d( self.inplanes,planes * block.expansion,kernel_size=1, stride=stride, output_padding=stride-1, bias=False),nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample))
        for i in range(1, blocks):
            layers.append(block(planes, planes))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.original_inplanes, 1, 1)
        x = self.upsample7(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.upsample2(x)
        x = self.conv1(x)
        x = self.bn1(x)

        return x

class ResnetFragmentEncoder(nn.Module):

    def __init__(self, inplanes, block, layers):
        super(ResnetFragmentEncoder, self).__init__()

        self.inplanes = inplanes
        self.block = block
        self.layers =  layers

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        return x

class LinearEncoders(nn.Module):
    def linearity(self):
        return nn.Sequential(nn.Linear(self.n, self.n), nn.Linear(self.n, self.n), nn.Linear(self.n, self.z))

    def __init__(self, z, n, c): # z for latent dimension, n for complexity of fc, c for clusters

        super(LinearEncoders, self).__init__()
        self.z = z
        self.n = n
        self.c = c
        self.fc_mu = nn.ModuleList([self.linearity() for i in range(c)])
        self.fc_var = nn.ModuleList([self.linearity() for i in range(c)])
        return

    def forward(self, datum):
        mu_list = []
        std_list = []

        for i in range(self.c):
            mu_list.append(self.fc_mu[i](datum))
            std_list.append(self.fc_var[i](datum))

        return mu_list, std_list

class LinearDecoders(nn.Module):
    def linearity(self):
        return nn.Sequential(nn.Linear(self.z*self.c, self.n), nn.Linear(self.n, self.n), nn.Linear(self.n, self.n))

    def __init__(self, z, n, c): # z for latent dimension, n for complexity of fc, c for clusters
        super(LinearDecoders, self).__init__()
        self.z = z
        self.n = n
        self.c = c
        self.dfc = nn.ModuleList([self.linearity()])
        return

    def forward(self, datum):
        return self.dfc[0](datum)

class DiscriminatorFC(nn.Module):
    def linearity(self):
        return nn.Sequential(nn.Linear(self.n, self.n), nn.Linear(self.n, self.n), nn.Linear(self.n, 2))

    def __init__(self, n, c): # z for latent dimension, n for complexity of fc, c for clusters

        super(DiscriminatorFC, self).__init__()
        self.n = n
        self.c = c
        self.fc = nn.ModuleList([self.linearity()])
        return

    def forward(self, datum):
        return self.fc[0](datum)

class RelationPrediction(nn.Module):
    def __init__(self, n, z, c, streams):
        super(RelationPrediction, self).__init__()
        self.n = n
        self.z = z
        self.c = c
        self.streams = streams
        self.full_FC = nn.Sequential(nn.Linear(self.z*self.c*self.streams, self.n), nn.Linear(self.n, self.n), nn.Linear(self.n, 16))

    def forward(self, datum):
        z_vectors = []
        for s in range(self.streams):
            z_vectors.append(datum[s])
        z_vector_cat = torch.cat(z_vectors,1)
        class_distribution = self.full_FC(z_vector_cat)
        return class_distribution

class MnistDiscriminator(nn.Module):
    def __init__(self, complexity="Complexity A", clusters=3, streams=2):

        super(MnistDiscriminator, self).__init__()
        self.complexity = complexity
        self.clusters = clusters
        self.streams = streams
        self.discriminate = nn.ModuleList([ResnetFragmentEncoder(inplanes=64, block=BasicBlock, layers=[2, 2, 2, 2]),DiscriminatorFC(512, self.clusters)])
        self.fake = False

    def forward(self, datums, stream_outputs, stream_inputs, stream_recombined):
        #
        discriminator_predictions = [None for i in range(len(datums))]
        discriminator_labels = [torch.ones(datums[0].size()[0]).long() for i in range(len(datums))] # real targets are 1s, fake targets are 0s

        discriminator_predictions_recombined = None
        discriminator_labels_recombined = torch.ones(datums[0].size()[0]).long() # real targets are 1s, fake targets are 0s

        if self.fake:  # Fake samples, 0 targets
            for stream in range(self.streams):
                discriminator_cnn = self.discriminate[0](stream_outputs[stream]).view(-1, 512 * 1 * 1)
                discriminator_predictions[stream] = self.discriminate[1](discriminator_cnn)
                discriminator_labels[stream] = torch.mul(discriminator_labels[stream], 0)

            discriminator_cnn = self.discriminate[0](stream_recombined).view(-1, 512 * 1 * 1)
            discriminator_predictions_recombined = self.discriminate[1](discriminator_cnn)
            discriminator_labels_recombined = torch.mul(discriminator_labels[stream], 0)
        else:
            for stream in range(self.streams):
                discriminator_cnn = self.discriminate[0](stream_inputs[stream]).view(-1, 512 * 1 * 1)
                discriminator_predictions[stream] = self.discriminate[1](discriminator_cnn)

        return discriminator_predictions, discriminator_labels, discriminator_predictions_recombined, discriminator_labels_recombined

class RecombinedGAN(nn.Module):

    def __init__(self, complexity="Complexity A", proper_size=(224,224)):
        super(RecombinedGAN, self).__init__()
        self.complexity = complexity
        self.proper_size = proper_size
        self.GAN = nn.ModuleList([MnistVAEMixture(complexity=self.complexity, proper_size=self.proper_size), MnistDiscriminator(self.complexity)])

    def forward(self, datums):
        stream_outputs, stream_recons, stream_inputs, stream_recombined, mu_lists, std_lists, z_sample_lists, class_predictions = self.GAN[0](datums)
        d_pred, d_labels = self.GAN[1](datums, stream_outputs, stream_inputs, stream_recombined)
        return stream_outputs, stream_recons, stream_inputs, mu_lists, std_lists, z_sample_lists, class_predictions, d_pred, d_labels

class MnistVAEMixture(nn.Module):

    def sample_z(self, mu, log_var):

        # Using reparameterization trick to sample from a gaussian mixture
        if self.inference:
            return mu
        eps = Variable(torch.randn(mu.size()[0], self.z)).cuda()
        return mu + torch.exp(log_var / 2) * eps

    def __init__(self, complexity="Complexity A", clusters=3, streams = 2, proper_size=(224,224)):
        super(MnistVAEMixture, self).__init__()
        self.inference = False
        self.complexity = complexity
        self.clusters = clusters
        self.streams = streams
        self.proper_size = proper_size

        self.sigmoid_in = nn.Sigmoid()
        self.sigmoid_out = nn.Sigmoid()
        self.tanh_in = nn.Tanh()

        if self.complexity == "Complexity A":
            self.z = 8
            self.encoder = nn.ModuleList([ResnetFragmentEncoder(inplanes=64, block=BasicBlock, layers=[2,2,2,2])])
            self.encoder_fc = nn.ModuleList([LinearEncoders(self.z, 512, self.clusters)])
            self.decoder_fc = nn.ModuleList([LinearDecoders(self.z, 512, self.clusters)])
            self.decoder = nn.ModuleList([ResnetFragmentDecoder(inplanes=512, block=ReverseBasicBlock, layers=[2,2,2,2])])
            self.classify = nn.ModuleList([RelationPrediction(512, self.z*self.clusters, 1, self.streams)])

        self.tanh_out = nn.Tanh()
        self.upsample = nn.Upsample(scale_factor=8)

    def forward(self, datums):
        # Proper input for face is 1 x 224 x 224

        stream_inputs = [None for i in range(len(datums))]
        stream_h_cat = [None for i in range(len(datums))]
        stream_outputs = [None for i in range(len(datums))]
        stream_recons = [None for i in range(len(datums))]

        mu_lists = [[] for i in range(len(datums))]
        std_lists = [[] for i in range(len(datums))]
        z_sample_lists = [[] for i in range(len(datums))]

        for stream_idx, stream in enumerate(datums):

            stream = self.upsample(stream)
            stream_in = self.tanh_in(stream)
            stream_inputs[stream_idx] = stream_in
            stream_h = self.encoder[0](stream_in)

            stream_h = stream_h.view(-1,512*1*1)
            mu_list, std_list = self.encoder_fc[0](stream_h)
            mu_lists[stream_idx] = mu_list # list of clusters containing mu vectors
            std_lists[stream_idx] = std_list
            z_sample_list = []

            for i in range(self.clusters):
                z_sample_list.append(self.sample_z(mu_list[i], std_list[i]))
            z_sample_lists[stream_idx] = z_sample_list
            # z_sample_list are the separate samples of z for the different clustering VAEs
            # combine them together, to form stream_h_cat which reconstructs a single image
            stream_h_cat[stream_idx] = torch.cat(z_sample_lists[stream_idx],1)
            stream_h_list = self.decoder_fc[0](stream_h_cat[stream_idx])
            stream_out = self.decoder[0](stream_h_list)
            stream_recons[stream_idx] = stream_out
            stream_ts = self.tanh_out(stream_out)
            stream_outputs[stream_idx]= stream_ts

        # Create a recombined sample for each pair in the batch (assume pair of 2 streams)
        z_pair_A = z_sample_lists[0]
        z_pair_B = z_sample_lists[1]
        random_cluster = random.randint(self.clusters)
        z_Recombined = [torch.zeros(self.z) for k in range(self.clusters)]
        for c in range(self.clusters):
            if random_cluster == c:
                z_Recombined[c] = torch.add(z_Recombined[c], z_pair_A[c])
            else:
                z_Recombined[c] = torch.add(z_Recombined[c], z_pair_B[c])

        stream_h_cat_recombined = torch.cat(z_Recombined, 1)
        stream_h_list = self.decoder_fc[0](stream_h_cat_recombined)
        stream_recombined = self.decoder[0](stream_h_list)
        stream_recons_recombined = stream_recombined
        stream_ts_recombined = self.tanh_out(stream_recons_recombined)
        stream_recombined = stream_ts_recombined

        class_prediction = self.classify[0](stream_h_cat)

        return stream_outputs, stream_recons, stream_inputs, stream_recombined, mu_lists, std_lists, z_sample_lists, class_prediction

    def decode(self, z_sample_lists):
        # input: a list of latent z's for each cluster
        stream_h_cat = torch.cat(z_sample_lists, 1)
        stream_h_list = self.decoder_fc[0](stream_h_cat)
        stream_out = self.decoder[0](stream_h_list)
        stream_recons = stream_out
        stream_ts = self.tanh_out(stream_out)
        stream_outputs = stream_ts
        return stream_outputs, stream_recons, z_sample_lists
