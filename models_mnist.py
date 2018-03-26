import torch
import torch.nn as nn

from torch.autograd import Variable
import pathlib
from sklearn.cluster import KMeans
import progressbar
import resnet_models
import torch.optim as optim
import math
import random
import visdom
import math
import numpy as np
import torch.nn.functional as F
from resnet_models import BasicBlock, ReverseBasicBlock

from torchvision.utils import save_image as save

vis_main = visdom.Visdom(env='main')
vis_plots = visdom.Visdom(env='plots')
vis_kernels = visdom.Visdom(env='kernels')
vis_class = visdom.Visdom(env='class')
vis_class_disc = visdom.Visdom(env='class_disc')
vis_rec = visdom.Visdom(env='rec')
vis_sample = visdom.Visdom(env='sample')
#################### Vanilla Model Double Stream

schedulers_types = {'MNISTVAE Zodiac':{'epochs' : 100,'lr_epoch' : 100000,'lr_mod' : 1.0,'lr_change' : 1.0,'lr_base' : 1e-3,'wd' : 0.0}}
complexity_type = {'MNISTVAE Zodiac': 'Complexity A'}

########################

model_choice = 'MNISTVAE Zodiac' # Build Scheduler for training and Parameters for model architecture

# For tomorrow, make a Resnet that encodes and decodes a sample and run it once. That is all.

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
        self.relu = nn.ReLU(inplace=True)
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
        self.relu = nn.ReLU(inplace=True)
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

    def forward(self, datums, stream_outputs, stream_inputs):
        #
        discriminator_predictions = [None for i in range(len(datums))]
        discriminator_labels = [torch.ones(datums[0].size()[0]).long() for i in range(len(datums))]
        if self.fake:  # Fake samples, 0
            for stream in range(self.streams):
                discriminator_cnn = self.discriminate[0](stream_outputs[stream]).view(-1, 512 * 1 * 1)
                discriminator_predictions[stream] = self.discriminate[1](discriminator_cnn)
                discriminator_labels[stream] = torch.mul(discriminator_labels[stream], 0)
        else:
            for stream in range(self.streams):
                discriminator_cnn = self.discriminate[0](stream_inputs[stream]).view(-1, 512 * 1 * 1)
                discriminator_predictions[stream] = self.discriminate[1](discriminator_cnn)

        return discriminator_predictions, discriminator_labels

class MnistVAEMixture(nn.Module):

    def sample_z(self, mu, log_var):

        # Using reparameterization trick to sample from a gaussian mixture
        if self.inference:
            return mu
        eps = Variable(torch.randn(mu.size()[0], self.z)).cuda()
        return mu + torch.exp(log_var / 2) * eps

    def __init__(self, complexity="Complexity A", clusters=3, streams = 2):
        super(MnistVAEMixture, self).__init__()
        self.inference = False
        self.complexity = complexity
        self.clusters = clusters
        self.streams = streams

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

        class_prediction = self.classify[0](stream_h_cat)

        return stream_outputs, stream_recons, stream_inputs, mu_lists, std_lists, z_sample_lists, class_prediction

    def decode(self, z_sample_lists):
        # input: a list of latent z's for each cluster
        stream_h_cat = torch.cat(z_sample_lists, 0)
        stream_h_list = self.decoder_fc[0](stream_h_cat)
        stream_out = self.decoder[0](stream_h_list)
        stream_recons = stream_out
        stream_ts = self.tanh_out(stream_out)
        stream_outputs = stream_ts
        return stream_outputs, stream_recons, z_sample_lists

class VAE_Building():

    def load_model(self, epoch=1000):
        self.VAE.load_state_dict(torch.load(self.directory_models + "VAE_model_" + str(epoch) + ".model"))
        self.Discriminator.load_state_dict(torch.load(self.directory_models + "Discriminator_model_" + str(epoch) + ".model"))
        self.VAE.inference = False

    def save_model(self, epoch=0):
        print(self.directory_models+"VAE_model_" + str(epoch) + ".model")
        torch.save(self.VAE.state_dict(), self.directory_models + "VAE_model_" + str(epoch) + ".model")
        torch.save(self.Discriminator.state_dict(), self.directory_models + "Discriminator_model_" + str(epoch) + ".model")

    def __init__(self, save_models=True, model_choice="X", dbs=None, result_path=None):

        self.inference = False
        self.save_models = save_models
        self.model_choice = model_choice
        self.dbs = dbs
        self.result_path = result_path

        self.optimizer = None

        self.main_directory = "MNIST VAE Lab/"
        self.result_path = result_path
        self.scheduler = schedulers_types[self.model_choice]

        ### Directories and Logging Definition

        self.root_directory = self.result_path
        self.version_directory = self.model_choice

        self.directory = self.root_directory + self.main_directory + self.version_directory
        print("Using directory : {}".format(self.directory))
        pathlib.Path(self.directory).mkdir(parents=True, exist_ok=True)

        self.directory_models = self.directory + "/models/"
        self.directory_visuals = self.directory + "/visuals/"

        pathlib.Path(self.directory_models).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.directory_visuals).mkdir(parents=True, exist_ok=True)

        self.VAE = MnistVAEMixture(complexity_type[self.model_choice]).cuda()
        self.Discriminator = MnistDiscriminator(complexity_type[self.model_choice]).cuda()

    def train(self):

        # Training hyper-parameters
        self.VAE.inference = False

        lr_epoch = self.scheduler['lr_epoch']
        lr_mod = self.scheduler['lr_mod']
        lr_change = self.scheduler['lr_change']
        lr_base = self.scheduler['lr_base']
        wd = self.scheduler['wd']
        epochs = self.scheduler['epochs']

        starting_epoch = 1
        epoch_save_all = 20
        epoch_test_all = 1

        if starting_epoch != 1:
            self.directory_models = self.directory + "/models/"
            self.VAE.load_state_dict(torch.load(self.directory_models + "VAE_model_" + str(starting_epoch - 1) + ".model"))
            self.Discriminator.load_state_dict(torch.load(self.directory_models + "Discriminator_model_" + str(starting_epoch - 1) + ".model"))

        self.optimizer_VAE = optim.Adam(self.VAE.parameters(), weight_decay=wd, lr=lr_base * lr_mod)
        self.optimizer_Discriminator = optim.Adam(self.Discriminator.parameters(), weight_decay=wd, lr=lr_base * lr_mod * 0.1)

        # Start training , testing and visualizing
        epoch_counting = {'train':[], 'test':[]}
        epoch_plot_acc = {'train':{'acc class':[],'acc dsc':[]}, 'test':{'acc class':[],'acc dsc':[]}}
        epoch_plot_loss = {'train':{'loss class':[],'loss dsc':[], 'loss mse':[],'loss kld':[],'loss gen':[]}, 'test':{'loss class':[],'loss dsc':[], 'loss mse':[],'loss kld':[],'loss gen':[]}}

        batch_counting = {'train':[]}
        batches_plot_loss = {'train':{'loss class':[],'loss dsc':[], 'loss mse':[],'loss kld':[],'loss gen':[]}}
        batches_plot_acc = {'train':{'acc class':[],'acc dsc':[]}}

        for epoch in range(starting_epoch, epochs + 1):
            epoch_counting['train'].append(epoch)

            print("Epoch {}".format(epoch))
            if epoch % lr_epoch == 0:
                lr_mod *= lr_change
                self.optimizer_VAE = optim.Adam(self.VAE.parameters(), weight_decay=wd, lr=lr_base * lr_mod)
                self.optimizer_Discriminator = optim.Adam(self.Discriminator.parameters(), weight_decay=wd, lr=lr_base * lr_mod * 0.1)

            print("TRAINING ----------------- epoch {}".format(epoch))
            self.inference = False
            self.VAE.train()
            self.Discriminator.train()

            bar = progressbar.ProgressBar()
            with progressbar.ProgressBar(max_value=len(self.dbs['train'])) as bar:
                correct_class = 0
                correct_discriminator = 0
                total_batch = 0

                epoch_plot_loss['train']['loss class'].append(0)
                epoch_plot_loss['train']['loss kld'].append(0)
                epoch_plot_loss['train']['loss mse'].append(0)
                epoch_plot_loss['train']['loss gen'].append(0)
                epoch_plot_loss['train']['loss dsc'].append(0)

                for i, datum in enumerate(self.dbs['train']):
                    self.Discriminator.fake = not self.Discriminator.fake

                    batch_counting['train'].append(i+(epoch-1)*len(self.dbs['train']))
                    in_datums = [Variable(datum["Image Pairs"][0].cuda()), Variable(datum["Image Pairs"][1].cuda())]
                    stream_outputs, stream_recons, stream_inputs, mu_lists, std_lists, z_sample_lists, class_predictions = self.VAE(in_datums)
                    d_pred, d_labels = self.Discriminator(in_datums, stream_outputs, stream_inputs)

                    for s in range(len(in_datums)):
                        vis_rec.image(torch.clamp(self.dbs['train'].ut(stream_inputs[s][0].cpu().data),0,1),  opts={'caption':"Train Rec"})
                        vis_rec.image(torch.clamp(self.dbs['train'].ut(stream_outputs[s][0].cpu().data),0,1),  opts={'caption':"Train Rec"})
                        if self.Discriminator.fake:
                            vis_class_disc.image(torch.clamp(self.dbs['train'].ut(stream_inputs[s][0].cpu().data),0,1), opts={'caption':'target {}, pred {}'.format(d_labels[s][0],d_pred[s][0].max(0)[1].cpu().data.numpy()[0])})
                        else:
                            vis_class_disc.image(torch.clamp(self.dbs['train'].ut(stream_outputs[s][0].cpu().data),0,1), opts={'caption':'target {}, pred {}'.format(d_labels[s][0],d_pred[s][0].max(0)[1].cpu().data.numpy()[0])})

                    for s in range(len(in_datums)):
                        vis_class.image(torch.clamp(self.dbs['train'].ut(stream_inputs[s][0].cpu().data),0,1), opts={'caption':'target {}/2, pred {}/2'.format(datum["Diff Labels"][0]+1,1+class_predictions[0].max(0)[1].cpu().data.numpy()[0])})

                    L_REC, L_KLD, L_class, L_Gen, L_Dsc = self.total_loss(i, stream_outputs, stream_inputs, mu_lists, std_lists, class_predictions, Variable(datum["Diff Labels"].cuda()), d_pred, d_labels)

                    batches_plot_loss['train']['loss class'].append(np.double(L_class.cpu().data.numpy()))
                    batches_plot_loss['train']['loss kld'].append(np.double(L_KLD.cpu().data.numpy()))
                    batches_plot_loss['train']['loss mse'].append(np.double(L_REC.cpu().data.numpy()))
                    batches_plot_loss['train']['loss gen'].append(np.double(L_Gen.cpu().data.numpy()))
                    batches_plot_loss['train']['loss dsc'].append(np.double(L_Dsc.cpu().data.numpy()))

                    epoch_plot_loss['train']['loss class'][epoch-starting_epoch] += (np.double(L_class.cpu().data.numpy()))
                    epoch_plot_loss['train']['loss kld'][epoch-starting_epoch] += (np.double(L_KLD.cpu().data.numpy()))
                    epoch_plot_loss['train']['loss mse'][epoch-starting_epoch] +=(np.double(L_REC.cpu().data.numpy()))
                    epoch_plot_loss['train']['loss gen'][epoch-starting_epoch] +=(np.double(L_Gen.cpu().data.numpy()))
                    epoch_plot_loss['train']['loss dsc'][epoch-starting_epoch] +=(np.double(L_Dsc.cpu().data.numpy()))

                    L_Generator = sum([L_REC, L_KLD, L_class, L_Gen])
                    L_Discriminator = L_Dsc

                    class_targets = Variable(datum["Diff Labels"])
                    predictions = class_predictions.max(1)[1].type_as(class_targets)
                    correct = np.double(predictions.eq(class_targets).data.numpy())
                    correct_class += (correct.sum())
                    total_batch += in_datums[0].size()[0]
                    batches_plot_acc['train']['acc class'].append(100*np.double(correct.sum())/ in_datums[0].size()[0])

                    current_correct_sum = 0
                    print(correct_discriminator, 2*total_batch)
                    for stream in range(len(stream_outputs)):
                        class_targets = Variable(d_labels[stream])
                        predictions = d_pred[stream].max(1)[1].type_as(class_targets)
                        correct = np.double(predictions.eq(class_targets).data.numpy())
                        correct_discriminator += correct.sum()
                        current_correct_sum += correct.sum()


                    batches_plot_acc['train']['acc dsc'].append(100*current_correct_sum/ (2*(in_datums[0].size()[0])))

                    if i == 0 and epoch == 1:
                        L_Generator.backward(retain_graph=True)
                        L_Discriminator.backward()
                    else: # Visualize plots for non-trained first epoch (starting random point)

                        self.optimizer_VAE.zero_grad()
                        L_Generator.backward(retain_graph=True)
                        my_dict = self.VAE.state_dict(keep_vars=True)
                        print("")
                        for param in my_dict.keys():
                            if 'weight' in param or 'bias' in param:
                                my_max = torch.max(my_dict[param].grad).cpu().data.numpy()[0]
                                my_min = torch.min(torch.abs(my_dict[param].grad)).cpu().data.numpy()[0]
                                if my_max > 1e3 or my_min < 1e-20:
                                    print("Exploding or vanishing gradient in: {}, min {:0.2f}, max {:0.2f}".format(param, my_min, my_max))

                        self.optimizer_VAE.step()

                        self.optimizer_Discriminator.zero_grad()
                        L_Discriminator.backward()
                        my_dict = self.Discriminator.state_dict(keep_vars=True)
                        for param in my_dict.keys():
                            if 'weight' in param or 'bias' in param:
                                my_max = torch.max(my_dict[param].grad).cpu().data.numpy()[0]
                                my_min = torch.min(torch.abs(my_dict[param].grad)).cpu().data.numpy()[0]
                                if my_max > 1e3 or my_min < 1e-20:
                                    print("Exploding or vanishing gradient in: {}, min {:0.2f}, max {:0.2f}".format(param, my_min, my_max))
                        self.optimizer_Discriminator.step()

                    bar.update(i)

                    ### Plot visualizations
                    trace = dict(x=batch_counting['train'], y=batches_plot_acc['train']['acc class'], mode="markers+lines", type='custom',marker = {'color': 'red', 'symbol': 0 , 'size': "5"})
                    layout = dict(title="Classification Accuracy Plot (train)", xaxis={'title': 'batch idx'}, yaxis={'title': 'class accuracy'})
                    vis_plots._send({'data': [trace], 'layout': layout, 'win': 'Class accuracy, training'})

                    trace = dict(x=batch_counting['train'], y=batches_plot_acc['train']['acc dsc'], mode="markers+lines", type='custom',marker = {'color': 'red', 'symbol': 0, 'size': "5"})
                    layout = dict(title="Discriminator Accuracy Plot (train)", xaxis={'title': 'batch idx'}, yaxis={'title': 'discriminator accuracy'})
                    vis_plots._send({'data': [trace], 'layout': layout, 'win': 'Discriminator accuracy, training'})

                    trace1 = dict(x=batch_counting['train'], y=batches_plot_loss['train']['loss dsc'], mode="markers+lines", type='custom',marker = {'color': 'red', 'symbol': 0, 'size': "5"}, name='disc')
                    trace2 = dict(x=batch_counting['train'], y=batches_plot_loss['train']['loss gen'], mode="markers+lines", type='custom',marker = {'color': 'blue', 'symbol': 0, 'size': "5"},name='gen')
                    trace3 = dict(x=batch_counting['train'], y=batches_plot_loss['train']['loss class'], mode="markers+lines", type='custom',marker = {'color': 'cyan', 'symbol': 0, 'size': "5"},name='class')
                    trace4 = dict(x=batch_counting['train'], y=batches_plot_loss['train']['loss mse'], mode="markers+lines", type='custom',marker = {'color': 'purple', 'symbol': 0, 'size': "5"},name='mse')
                    trace5 = dict(x=batch_counting['train'], y=batches_plot_loss['train']['loss kld'], mode="markers+lines", type='custom',marker = {'color': 'black', 'symbol': 0, 'size': "5"},name='kld')
                    layout = dict(title="Losses Plot (train)", xaxis={'title': 'batch idx'}, yaxis={'title': 'All losses'})
                    vis_plots._send({'data': [trace1, trace2, trace3, trace4, trace5], 'layout': layout, 'win': 'All losses, training'})

            epoch_plot_acc['train']['acc class'].append(correct_class*100.0/total_batch)
            trace = dict(x=epoch_counting['train'], y=epoch_plot_acc['train']['acc class'], mode="markers+lines", type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
            layout = dict(title="Classification Accuracy Plot (train)", xaxis={'title': 'epoch idx'}, yaxis={'title': 'class accuracy'})
            vis_plots._send({'data': [trace], 'layout': layout, 'win': 'Class accuracy, training (epoch)'})

            epoch_plot_acc['train']['acc dsc'].append(correct_discriminator * 100/ (2*total_batch))
            trace = dict(x=epoch_counting['train'], y=epoch_plot_acc['train']['acc dsc'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
            layout = dict(title="Discriminator Accuracy Plot (train)", xaxis={'title': 'epoch idx'},yaxis={'title': 'class accuracy'})
            vis_plots._send({'data': [trace], 'layout': layout, 'win': 'Discriminator accuracy, training (epoch)'})

            trace1 = dict(x=epoch_counting['train'], y=epoch_plot_loss['train']['loss dsc'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"}, name='disc')
            trace2 = dict(x=epoch_counting['train'], y=epoch_plot_loss['train']['loss gen'], mode="markers+lines",type='custom', marker={'color': 'blue', 'symbol': 0, 'size': "5"}, name='gen')
            trace3 = dict(x=epoch_counting['train'], y=epoch_plot_loss['train']['loss class'], mode="markers+lines",type='custom', marker={'color': 'cyan', 'symbol': 0, 'size': "5"}, name='class')
            trace4 = dict(x=epoch_counting['train'], y=epoch_plot_loss['train']['loss mse'], mode="markers+lines",type='custom', marker={'color': 'purple', 'symbol': 0, 'size': "5"}, name='mse')
            trace5 = dict(x=epoch_counting['train'], y=epoch_plot_loss['train']['loss kld'], mode="markers+lines",type='custom', marker={'color': 'black', 'symbol': 0, 'size': "5"}, name='kld')
            layout = dict(title="Losses Plot (train)", xaxis={'title': 'epoch idx'}, yaxis={'title': 'All losses'})
            vis_plots._send({'data': [trace1, trace2, trace3, trace4, trace5], 'layout': layout, 'win': 'All losses, training (epoch)'})

            # END OF VISUALIZATIONS OF PLOTS

            # Create samples for Z from 0 vector
            self.VAE.eval()
            z_lists = [Variable(torch.zeros((self.VAE.z)), volatile=True).cuda() for i in range(self.VAE.clusters)]
            sample_output, _, _ = self.VAE.decode(z_lists)
            vis_sample.image(torch.clamp(self.dbs['train'].ut(sample_output[0].cpu().data),0,0.1),  opts={'caption':"Sample Rec"})

            # Vary samples of z
            for i in range(self.VAE.clusters):
                z_lists = [Variable(torch.zeros((self.VAE.z)), volatile=True).cuda() for i in range(self.VAE.clusters)]
                torch.add(z_lists[i], Variable(torch.normal(torch.zeros((self.VAE.z)),torch.ones(((self.VAE.z)))), volatile=True).cuda())
                sample_output, _, _ = self.VAE.decode(z_lists)
                vis_sample.image(torch.clamp(self.dbs['train'].ut(sample_output[0].cpu().data),0,1),  opts={'caption':"Sample Rec, for unit gaussian on cluster {}".format(i+1)})


            if epoch % epoch_test_all == 0:
                epoch_test = int(math.floor(epoch/epoch_test_all))
                epoch_counting['test'].append(epoch_test)
                print("TESTING -----------------")
                self.inference = True
                self.VAE.eval()
                self.Discriminator.eval()

                bar = progressbar.ProgressBar()
                with progressbar.ProgressBar(max_value=len(self.dbs['test'])) as bar:

                    correct_class = 0
                    correct_discriminator = 0
                    total_batch = 0

                    epoch_plot_loss['test']['loss class'].append(0)
                    epoch_plot_loss['test']['loss kld'].append(0)
                    epoch_plot_loss['test']['loss mse'].append(0)
                    epoch_plot_loss['test']['loss gen'].append(0)
                    epoch_plot_loss['test']['loss dsc'].append(0)

                    for i, datum in enumerate(self.dbs['test']):
                        self.Discriminator.fake = not self.Discriminator.fake

                        in_datums = [Variable(datum["Image Pairs"][0].cuda(), volatile=True), Variable(datum["Image Pairs"][1].cuda(), volatile=True)]
                        stream_outputs, stream_recons, stream_inputs, mu_lists, std_lists, z_sample_lists, class_predictions = self.VAE(in_datums)
                        d_pred, d_labels = self.Discriminator(in_datums, stream_outputs, stream_inputs)

                        for s in range(len(in_datums)):
                            vis_rec.image(torch.clamp(self.dbs['test'].ut(stream_inputs[s][0].cpu().data),0,1), opts={'caption':"Test Rec"})
                            vis_rec.image(torch.clamp(self.dbs['test'].ut(stream_outputs[s][0].cpu().data),0,1),  opts={'caption':"Test Rec"})

                            if self.Discriminator.fake:
                                vis_class_disc.image(torch.clamp(self.dbs['train'].ut(stream_inputs[s][0].cpu().data), 0, 1), opts={'caption': 'Test target {}, pred {}'.format(d_labels[s][0], d_pred[s][0].max(0)[1].cpu().data.numpy()[0])})
                            else:
                                vis_class_disc.image(torch.clamp(self.dbs['train'].ut(stream_outputs[s][0].cpu().data), 0, 1), opts={'caption': 'Test target {}, pred {}'.format(d_labels[s][0], d_pred[s][0].max(0)[1].cpu().data.numpy()[0])})

                        for s in range(len(in_datums)):
                            vis_class.image(torch.clamp(self.dbs['train'].ut(stream_inputs[s][0].cpu().data), 0, 1), opts={'caption': 'Test target {}/2, pred {}/2'.format(datum["Diff Labels"][0] + 1, 1 + class_predictions[0].max(0)[1].cpu().data.numpy()[0])})

                        class_targets = Variable(datum["Diff Labels"], volatile=True).cuda()

                        L_REC, L_KLD, L_class, L_Gen, L_Dsc = self.total_loss(i, stream_outputs, stream_inputs,mu_lists, std_lists, class_predictions,Variable(datum["Diff Labels"].cuda(), volatile=True),d_pred, d_labels)

                        epoch_plot_loss['test']['loss class'][epoch_test-starting_epoch] += (np.double(L_class.cpu().data.numpy()))
                        epoch_plot_loss['test']['loss kld'][epoch_test-starting_epoch] += (np.double(L_KLD.cpu().data.numpy()))
                        epoch_plot_loss['test']['loss mse'][epoch_test-starting_epoch] += (np.double(L_REC.cpu().data.numpy()))
                        epoch_plot_loss['test']['loss gen'][epoch_test-starting_epoch] += (np.double(L_Gen.cpu().data.numpy()))
                        epoch_plot_loss['test']['loss dsc'][epoch_test-starting_epoch] += (np.double(L_Dsc.cpu().data.numpy()))

                        class_targets = Variable(datum["Diff Labels"], volatile=True)
                        predictions = class_predictions.max(1)[1].type_as(class_targets)
                        correct = np.double(predictions.eq(class_targets).data.numpy())
                        correct_class += (correct.sum())
                        total_batch += in_datums[0].size()[0]

                        current_correct_sum = 0
                        for stream in range(len(stream_outputs)):
                            class_targets = Variable(d_labels[stream])
                            predictions = d_pred[stream].max(1)[1].type_as(class_targets)
                            correct = np.double(predictions.eq(class_targets).data.numpy())
                            correct_discriminator += correct.sum()

                        bar.update(i)
                epoch_plot_acc['test']['acc class'].append(correct_class * 100.0 / total_batch)
                trace = dict(x=epoch_counting['test'], y=epoch_plot_acc['test']['acc class'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                layout = dict(title="Classification Accuracy Plot (test)", xaxis={'title': 'epoch idx'},yaxis={'title': 'class accuracy'})
                vis_plots._send({'data': [trace], 'layout': layout, 'win': 'Class accuracy, testing (epoch)'})

                epoch_plot_acc['test']['acc dsc'].append(correct_discriminator * 100 / (2 * total_batch))
                trace = dict(x=epoch_counting['test'], y=epoch_plot_acc['test']['acc dsc'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                layout = dict(title="Discriminator Accuracy Plot (test)", xaxis={'title': 'epoch idx'},yaxis={'title': 'class accuracy'})
                vis_plots._send({'data': [trace], 'layout': layout, 'win': 'Discriminator accuracy, testing (epoch)'})

                trace1 = dict(x=epoch_counting['test'], y=epoch_plot_loss['test']['loss dsc'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"}, name='disc')
                trace2 = dict(x=epoch_counting['test'], y=epoch_plot_loss['test']['loss gen'], mode="markers+lines",type='custom', marker={'color': 'blue', 'symbol': 0, 'size': "5"}, name='gen')
                trace3 = dict(x=epoch_counting['test'], y=epoch_plot_loss['test']['loss class'], mode="markers+lines",type='custom', marker={'color': 'cyan', 'symbol': 0, 'size': "5"}, name='class')
                trace4 = dict(x=epoch_counting['test'], y=epoch_plot_loss['test']['loss mse'], mode="markers+lines",type='custom', marker={'color': 'purple', 'symbol': 0, 'size': "5"}, name='mse')
                trace5 = dict(x=epoch_counting['test'], y=epoch_plot_loss['test']['loss kld'], mode="markers+lines",type='custom', marker={'color': 'black', 'symbol': 0, 'size': "5"}, name='kld')
                layout = dict(title="Losses Plot (test)", xaxis={'title': 'epoch idx'}, yaxis={'title': 'All losses'})
                vis_plots._send({'data': [trace1, trace2, trace3, trace4, trace5], 'layout': layout,'win': 'All losses, testing (epoch)'})

            if epoch % epoch_save_all == 0 and epoch != starting_epoch:
                print("SAVING MODEL NOW epoch:{}".format(epoch))
                if self.save_models:
                    self.save_model(epoch=epoch)


    def total_loss(self, batch_idx, recon_x, x, mu, logvar, class_pred, target, d_pred, d_labels):

        L_REC = 0
        L_KLD = 0
        L_class = 0
        L_Gen = 0
        L_Dsc = 0

        for stream_idx in range(len(recon_x)):
            stream_in = x[stream_idx]
            stream_out = recon_x[stream_idx]
            mu_cat = torch.cat(mu[stream_idx],dim=1)
            logvar_cat = torch.cat(logvar[stream_idx],dim=1)
            L_REC += F.mse_loss(stream_out.view(-1, 1 * 224 *224),stream_in.view(-1, 1 * 224* 224))
            L_KLD += -0.5 * torch.sum(1 + logvar_cat - mu_cat.pow(2) - logvar_cat.exp())

        L_KLD /= recon_x[0].size()[0] * 224 * 224 * len(recon_x)
        L_KLD = torch.clamp(L_KLD, 0, 1000)

        criterion = nn.CrossEntropyLoss()
        L_class = criterion(class_pred, target.squeeze())

        for stream in range(len(recon_x)):
            L_Gen -= criterion(d_pred[stream], Variable(d_labels[stream].cuda()))
            L_Dsc += criterion(d_pred[stream], Variable(d_labels[stream].cuda()))

        #print("\nREC: ",L_REC.data.cpu().numpy(),"KLD: ", L_KLD.data.cpu().numpy(), "Class: ", L_class.data.cpu().numpy(), "Generator loss->", L_Gen.data.cpu().numpy(), "Discriminator loss->", L_Dsc.data.cpu().numpy())

        return L_REC, L_KLD, L_class, L_Gen, L_Dsc

########################
