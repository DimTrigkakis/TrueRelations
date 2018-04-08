import torch
import torch.nn as nn
import random
from torch.autograd import Variable
import math
from resnet_models import BasicBlock, ReverseBasicBlock

class ResnetFragmentDecoder(nn.Module):

    def __init__(self, inplanes, block, layers, channels):
        super(ResnetFragmentDecoder, self).__init__()

        self.original_inplanes = inplanes
        self.inplanes = inplanes
        self.block = block
        self.layers = layers
        self.channels = channels

        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample7 = nn.Upsample(scale_factor=7)
        self.relu = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.conv1 = nn.ConvTranspose2d(64, self.channels, kernel_size=7, stride=2, output_padding=1, padding=3,bias=False)

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

    def __init__(self, inplanes, block, layers, channels):
        super(ResnetFragmentEncoder, self).__init__()

        self.inplanes = inplanes
        self.block = block
        self.layers =  layers
        self.channels = channels

        self.conv1 = nn.Conv2d(self.channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
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
        return nn.Sequential(nn.Linear(self.n, self.h), nn.Linear(self.h, self.h), nn.Linear(self.h, self.z))

    def __init__(self, z, n, c, h=32): # z for latent dimension, n for complexity of fc, c for clusters

        super(LinearEncoders, self).__init__()
        self.z = z
        self.n = n
        self.c = c
        self.h = h
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
        return nn.Sequential(nn.Linear(self.z*self.c, self.h), nn.Linear(self.h, self.h), nn.Linear(self.h, self.n))

    def __init__(self, z, n, c, h=32): # z for latent dimension, n for complexity of fc, c for clusters
        super(LinearDecoders, self).__init__()
        self.z = z
        self.n = n
        self.c = c
        self.h = h
        self.dfc = nn.ModuleList([self.linearity()])
        return

    def forward(self, datum):
        return self.dfc[0](datum)

class DiscriminatorFC(nn.Module):
    def linearity(self):
        return nn.Sequential(nn.Linear(self.n, self.h), nn.Linear(self.h, self.h), nn.Linear(self.h, 2))

    def __init__(self, n, c, h): # z for latent dimension, n for complexity of fc, c for clusters

        super(DiscriminatorFC, self).__init__()
        self.n = n
        self.c = c
        self.h = h
        self.fc = nn.ModuleList([self.linearity()])
        return

    def forward(self, datum):
        return self.fc[0](datum)

class RelationPrediction(nn.Module):
    def __init__(self, z, c, h, streams, classes=16, domains=5):
        super(RelationPrediction, self).__init__()
        self.h = h
        self.z = z
        self.c = c
        self.streams = streams
        self.classes = classes
        self.domains = domains
        self.full_FC = nn.Sequential(nn.Linear(self.z*self.c*self.streams, self.h), nn.Linear(self.h, self.h), nn.Linear(self.h, self.classes))
        self.full_FC_domain = nn.Sequential(nn.Linear(self.z*self.c*self.streams, self.h), nn.Linear(self.h, self.h), nn.Linear(self.h, self.domains))

    def forward(self, datum):
        z_vectors = []
        for s in range(self.streams):
            z_vectors.append(datum[s])
        z_vector_cat = torch.cat(z_vectors,1)
        class_distribution = self.full_FC(z_vector_cat)
        class_distribution_domain = self.full_FC_domain(z_vector_cat)
        return class_distribution, class_distribution_domain

class FaceDiscriminator(nn.Module):
    def __init__(self, complexity="Complexity A", clusters=3, streams=4, channels=3):

        super(FaceDiscriminator, self).__init__()
        self.complexity = complexity
        self.clusters = clusters
        self.streams = streams
        self.channels = channels

        if self.complexity == "Complexity A":
            self.hidden = 128
            self.discriminate = nn.ModuleList([ResnetFragmentEncoder(inplanes=64, block=BasicBlock, layers=[2, 2, 2, 2], channels=self.channels),DiscriminatorFC(512, self.clusters, self.hidden)])

        self.fake = False

    def forward(self, datums, stream_outputs, stream_inputs, stream_recombined):
        #
        discriminator_predictions = [None for i in range(len(datums))]
        discriminator_labels = [torch.ones(datums[0].size()[0]).long() for i in range(len(datums))] # real targets are 1s, fake targets are 0s

        discriminator_predictions_recombined = [None for i in range(len(datums)//2)]
        discriminator_labels_recombined = [torch.ones(datums[0].size()[0]).long() for i in range(len(datums)//2)] # real targets are 1s, fake targets are 0s

        if self.fake:  # Fake samples, 0 targets
            #  Create 2 fake samples from recombination and 4 from reconstructions
            for stream in range(self.streams):
                discriminator_cnn = self.discriminate[0](stream_outputs[stream]).view(-1, 512 * 1 * 1)
                discriminator_predictions[stream] = self.discriminate[1](discriminator_cnn)
                discriminator_labels[stream] = torch.mul(discriminator_labels[stream], 0)
            for stream in range(self.streams//2):
                discriminator_cnn = self.discriminate[0](stream_recombined[stream]).view(-1, 512 * 1 * 1)
                discriminator_predictions_recombined[stream] = self.discriminate[1](discriminator_cnn)
                discriminator_labels_recombined[stream] = torch.mul(discriminator_labels[stream], 0)
        else:
            for stream in range(self.streams):
                discriminator_cnn = self.discriminate[0](stream_inputs[stream]).view(-1, 512 * 1 * 1)
                discriminator_predictions[stream] = self.discriminate[1](discriminator_cnn)

        return discriminator_predictions, discriminator_labels, discriminator_predictions_recombined, discriminator_labels_recombined

class RecombinedGAN(nn.Module):

    def __init__(self, complexity="Complexity A", proper_size=(224,224), channels = 3, hidden=512):
        super(RecombinedGAN, self).__init__()
        self.complexity = complexity
        self.proper_size = proper_size
        self.channels = channels
        self.hidden = hidden
        self.GAN = nn.ModuleList([FaceVAEMixture(complexity=self.complexity, proper_size=self.proper_size, channels=self.channels), FaceDiscriminator(complexity=self.complexity, channels=self.channels)])

    def forward(self, datums):
        stream_outputs, stream_recons, stream_inputs, stream_recombined, mu_lists, std_lists, z_sample_lists, class_predictions, domain_predictions = self.GAN[0](datums)
        d_pred, d_labels, d_pred_r, d_labels_r = self.GAN[1](datums, stream_outputs, stream_inputs, stream_recombined)
        return stream_outputs, stream_recons, stream_inputs, mu_lists, std_lists, z_sample_lists, class_predictions, domain_predictions, d_pred, d_labels, d_pred_r, d_labels_r

class FaceVAEMixture(nn.Module):

    def sample_z(self, mu, log_var):

        # Using reparameterization trick to sample from a gaussian mixture
        if self.inference:
            return mu
        eps = Variable(torch.randn(mu.size()[0], self.z)).cuda()
        return mu + torch.exp(log_var / 2) * eps

    def __init__(self, complexity="Complexity A", clusters=3, streams = 4, proper_size=(224,224), channels=3):
        super(FaceVAEMixture, self).__init__()
        self.inference = False
        self.complexity = complexity
        self.clusters = clusters
        self.streams = streams
        self.channels = channels
        self.proper_size = proper_size

        self.sigmoid_in = nn.Sigmoid()
        self.sigmoid_out = nn.Sigmoid()
        self.tanh_in = nn.Tanh()

        if self.complexity == "Complexity A":
            self.z = 8
            self.hidden = 128
            self.encoder = nn.ModuleList([ResnetFragmentEncoder(inplanes=64, block=BasicBlock, layers=[2,2,2,2], channels=self.channels)])
            self.encoder_fc = nn.ModuleList([LinearEncoders(self.z, 512, self.clusters, self.hidden)])
            self.decoder_fc = nn.ModuleList([LinearDecoders(self.z, 512, self.clusters, self.hidden)])
            self.decoder = nn.ModuleList([ResnetFragmentDecoder(inplanes=512, block=ReverseBasicBlock, layers=[2,2,2,2], channels=self.channels)])
            self.classify = nn.ModuleList([RelationPrediction(self.z, self.clusters, self.hidden, self.streams//2, classes=16, domains=5)])

        self.tanh_out = nn.Tanh()
        #self.upsample = nn.Upsample(scale_factor=8)

    def forward(self, datums):
        # Proper input for face is 3 x 224 x 224

        stream_inputs = [None for i in range(len(datums))]
        stream_h_cat = [None for i in range(len(datums))]
        stream_outputs = [None for i in range(len(datums))]
        stream_recons = [None for i in range(len(datums))]

        mu_lists = [[] for i in range(len(datums))]
        std_lists = [[] for i in range(len(datums))]
        z_sample_lists = [[] for i in range(len(datums))]

        for stream_idx, stream in enumerate(datums):
            #stream = self.upsample(stream)
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
        z_pair_random_A = z_sample_lists[2]
        z_pair_random_B = z_sample_lists[3]

        random_cluster = random.randint(0,self.clusters)
        z_Recombined = [Variable(torch.zeros(self.z)).cuda() for k in range(self.clusters)]
        for c in range(self.clusters):
            if random_cluster == c:
                z_Recombined[c] = torch.add(z_Recombined[c], z_pair_A[c])
            else:
                z_Recombined[c] = torch.add(z_Recombined[c], z_pair_random_A[c])

        rec = [] # Recombined list of streams of pairs

        r = torch.cat(z_Recombined, 1)
        r = self.decoder_fc[0](r)
        r = self.decoder[0](r)
        rec.append(self.tanh_out(r))

        random_cluster = random.randint(0, self.clusters)
        z_Recombined2 = [Variable(torch.zeros(self.z)).cuda() for k in range(self.clusters)]
        for c in range(self.clusters):
            if random_cluster == c:
                z_Recombined2[c] = torch.add(z_Recombined2[c], z_pair_B[c])
            else:
                z_Recombined2[c] = torch.add(z_Recombined2[c], z_pair_random_B[c])

        r2 = torch.cat(z_Recombined2, 1)
        r2 = self.decoder_fc[0](r2)
        r2 = self.decoder[0](r2)
        rec.append(self.tanh_out(r2))

        class_prediction, domain_prediction = self.classify[0]([stream_h_cat[0], stream_h_cat[2]]) # Classify the relationship based on the real faces

        return stream_outputs, stream_recons, stream_inputs, rec, mu_lists, std_lists, z_sample_lists, class_prediction, domain_prediction

    def decode(self, z_sample_lists):
        # input: a list of latent z's for each cluster
        stream_h_cat = torch.cat(z_sample_lists, 1)
        stream_h_list = self.decoder_fc[0](stream_h_cat)
        stream_out = self.decoder[0](stream_h_list)
        stream_recons = stream_out
        stream_ts = self.tanh_out(stream_out)
        stream_outputs = stream_ts
        return stream_outputs, stream_recons, z_sample_lists

