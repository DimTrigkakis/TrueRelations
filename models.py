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
import torch.nn.functional as F
from resnet_models import BasicBlock, ReverseBasicBlock

from torchvision.utils import save_image as save

vis = visdom.Visdom()
#################### Vanilla Model Double Stream

class StreamResnet(nn.ModuleList):
    def __init__(self, dbs = None, config=None):
        super(StreamResnet, self).__init__()
        self.dbs = dbs
        self.config = config

        myModuleList = [resnet_models.resnet34(pretrained=True) for i in range(self.config["streams"])]
        init_fc = self.config['streams']*1000.0
        self.streams = nn.ModuleList(myModuleList) # Are they optimized in self.parameters()????? DEBUG COMMENT
        self.fc = nn.Sequential(nn.Linear(int(init_fc), 4096), nn.Linear(4096,4096), nn.Linear(4096, 16))

        if self.config["optimizer"] == "Adam":
            self.config["optimizer"] = optim.Adam(self.parameters(), lr=1e-5,weight_decay=0.0)
        self.cuda()

    def forward(self, datums):
        feature_vectors = []
        for i in range(self.config['streams']):
            fv = self.streams[i](datums[i])
            feature_vectors.append(fv)

        # Concatenate the feature vector variables into a single feature vector
        final_feature_vector = torch.cat(feature_vectors,dim=1)
        final_output = self.fc(final_feature_vector)

        return final_output

    def __train__(self, db_type=None):
        self.train()
        db = self.dbs[db_type]
        bar = progressbar.ProgressBar()
        for epoch in range(self.config["epochs"]):

            with progressbar.ProgressBar(max_value=len(db.loader)) as bar:
                for i, datum in enumerate(db.loader):
                    self.config['optimizer'].zero_grad()
                    input_composure = [Variable(datum["FaceA"].cuda()),Variable(datum["FaceB"].cuda()), Variable(datum["BodyA"].cuda()),Variable(datum["BodyB"].cuda()), Variable(datum["Whole"].cuda())]
                    outputs = self(input_composure)
                    loss = self.config['criterion'](outputs, Variable(datum["Label"].cuda()).squeeze())
                    loss.backward()
                    self.config['optimizer'].step()
                    bar.update(i)

            print('[%d] Loss: %.3f' % (epoch + 1, loss.data[0]))

    def __acc__(self, db_type):
        total = 0
        correct = 0
        self.eval()

        with progressbar.ProgressBar(max_value=len(self.dbs[db_type].loader)) as bar:
            for i, datum in enumerate(self.dbs[db_type].loader):
                labels = Variable(datum["Label"].cuda())
                input_composure = [Variable(datum["FaceA"].cuda()), Variable(datum["FaceB"].cuda()),Variable(datum["BodyA"].cuda()), Variable(datum["BodyB"].cuda()),Variable(datum["Whole"].cuda())]

                outputs = self(input_composure)
                pos, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze().data).sum()
                bar.update(i)

        print('Accuracy of the network on the db {} images: {:.2f}% ({}/{})'.format(db_type, 100 * correct / total, correct, total))

    def perform(self,db_type="train", method="train"):
        if method == "train":
            self.__train__(db_type)
        elif method == "acc":
            self.__acc__(db_type)

######################## PROPER DOUBLE FACE VAEs

schedulers_types = {'FaceVAE Zodiac':{'epochs' : 10000,'lr_epoch' : 100000,'lr_mod' : 1.0,'lr_change' : 1.0,'lr_base' : 1e-4,'wd' : 0.0}}
complexity_type = {'FaceVAE Zodiac': 'Complexity A'}

########################

model_choice = 'FaceVAE Zodiac' # Build Scheduler for training and Parameters for model architecture

# For tomorrow, make a Resnet that encodes and decodes a sample and run it once. That is all.

class ResnetFragmentDecoder(nn.Module):

    def __init__(self, inplanes, block, layers):
        super(ResnetFragmentDecoder, self).__init__()

        self.original_inplanes = inplanes
        self.inplanes = inplanes
        self.block = block
        self.layers =  layers

        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample7 = nn.Upsample(scale_factor=7)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, output_padding=1, padding=3,bias=False)

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

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
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
        return nn.Sequential(nn.Linear(self.z, self.n), nn.Linear(self.n, self.n), nn.Linear(self.n, self.n))

    def __init__(self, z, n, c): # z for latent dimension, n for complexity of fc, c for clusters
        super(LinearDecoders, self).__init__()
        self.z = z
        self.n = n
        self.c = c
        self.dfc = nn.ModuleList([self.linearity() for i in range(c)])
        return

    def forward(self, datum):
        data_list = []

        for i in range(self.c):
            data_list.append(self.dfc[i](datum[i]))

        return data_list

class ClusterClassPrediction(nn.Module):
    def __init__(self, n, z, c, streams):
        super(ClusterClassPrediction, self).__init__()
        self.n = n
        self.z = z
        self.c = c
        self.streams = streams
        self.full_FC = nn.Sequential(nn.Linear(self.z*self.c*self.streams, self.n), nn.Linear(self.n, self.n), nn.Linear(self.n, 16))

    def forward(self, datum):
        z_vectors = []
        for s in range(self.streams):
            for c in range(self.c):
                z_vectors.append(datum[s][c])
        z_vector_cat = torch.cat(z_vectors,1)
        class_distribution = self.full_FC(z_vector_cat)
        return class_distribution

class FaceVAEMixture(nn.Module):

    def sample_z(self, mu, log_var):

        # Using reparameterization trick to sample from a gaussian mixture
        if self.inference:
            return mu
        eps = Variable(torch.randn(mu.size()[0], self.z)).cuda()
        return mu + torch.exp(log_var / 2) * eps

    def __init__(self, complexity="Complexity A", clusters=4):
        super(FaceVAEMixture, self).__init__()
        self.inference = False
        self.complexity = complexity
        self.clusters = 3
        self.streams = 2

        self.sigmoid_in = nn.Sigmoid()
        self.sigmoid_out = nn.Sigmoid()
        self.tanh_in = nn.Tanh()

        if self.complexity == "Complexity A":
            self.z = 32
            self.encoder = nn.ModuleList([ResnetFragmentEncoder(inplanes=64, block=BasicBlock, layers=[2,2,2,2])])
            self.encoder_fc = nn.ModuleList([LinearEncoders(self.z, 512, self.clusters)])
            self.decoder_fc = nn.ModuleList([LinearDecoders(self.z, 512, self.clusters)])
            self.decoder = nn.ModuleList([ResnetFragmentDecoder(inplanes=512, block=ReverseBasicBlock, layers=[2,2,2,2])])
            self.classify = nn.ModuleList([ClusterClassPrediction(512, self.z, self.clusters, self.streams)])

        self.tanh_out = nn.Tanh()

    def forward(self, datums):
        # Proper input for face is 3 x 224 x 224
        stream_outputs = [[] for i in range(len(datums))]
        stream_recons = [[] for i in range(len(datums))]
        mu_lists = [[] for i in range(len(datums))]
        std_lists = [[] for i in range(len(datums))]
        z_sample_lists = [[] for i in range(len(datums))]
        stream_inputs = []
        for stream_idx, stream in enumerate(datums):
            stream_in = self.tanh_in(stream)
            stream_inputs.append(stream_in)
            stream_h = self.encoder[0](stream_in)

            stream_h = stream_h.view(-1,512*1*1)
            mu_list, std_list = self.encoder_fc[0](stream_h)
            mu_lists[stream_idx]= mu_list
            std_lists[stream_idx] = std_list
            z_sample_list = []
            for i in range(self.clusters):
                z_sample_list.append(self.sample_z(mu_list[i], std_list[i]))
            z_sample_lists[stream_idx] = z_sample_list
            stream_h_list = self.decoder_fc[0](z_sample_list)
            for i in range(self.clusters):
                stream_out = self.decoder[0](stream_h_list[i])
                stream_recons[stream_idx].append(stream_out)
                stream_ts = self.tanh_out(stream_out)
                stream_outputs[stream_idx].append(stream_ts)

        class_prediction = self.classify[0](z_sample_lists)

        return stream_outputs, stream_recons, stream_inputs, mu_lists, std_lists, z_sample_lists, class_prediction

    def decode(self, z_sample):
        return None

class VAE_Building():

    def load_model(self, epoch=1000):
        self.VAE.load_state_dict(torch.load(self.directory_models + "model_" + str(epoch) + ".model"))
        self.VAE.inference = False

    def save_model(self, epoch=0):
        torch.save(self.VAE.state_dict(), self.directory_models + "model_" + str(epoch) + ".model")

    def __init__(self, save_models=True, model_choice="X", dbs=None, result_path=None):

        self.inference = False
        self.save_models = save_models
        self.model_choice = model_choice
        self.dbs = dbs
        self.result_path = result_path

        self.optimizer = None

        self.main_directory = "Face VAE Lab/"
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

        self.VAE = FaceVAEMixture(complexity_type[self.model_choice]).cuda()

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
        epoch_save_all = 100

        if starting_epoch != 1:
            self.directory_models = self.directory + "/models/"
            self.VAE.load_state_dict(torch.load(self.directory_models + "model_" + str(starting_epoch - 1) + ".model"))

        self.optimizer = optim.Adam(self.VAE.parameters(), weight_decay=wd, lr=lr_base * lr_mod)

        # Start training
        for epoch in range(starting_epoch, epochs + 1):

            self.inference = False
            print("Epoch {}".format(epoch))
            if epoch % lr_epoch == 0:
                lr_mod *= lr_change
                self.optimizer = optim.Adam(self.VAE.parameters(), weight_decay=wd, lr=lr_base * lr_mod)

            cluster = True
            if epoch % epoch_save_all != 0:
                if cluster:
                    # cluster
                    cluster_lists = []
                    for i, datum in enumerate(self.dbs['train'].loader):
                        in_datums = [Variable(datum["FaceA"].cuda()), Variable(datum["FaceB"].cuda())]
                        self.VAE.inference = True
                        stream_outputs, stream_recons, stream_inputs, mu_lists, std_lists, z_sample_lists, class_predictions = self.VAE(in_datums)
                        for j in range(datum["FaceA"].size()[0]):
                            cpu_z_list = [[],[]]
                            for k in range(3):
                                cpu_z_list[0].append(z_sample_lists[0][k][j].data.cpu())
                                cpu_z_list[1].append(z_sample_lists[1][k][j].data.cpu())
                            cluster_lists.append((datum["FaceA"][j], cpu_z_list[0], datum["Label"][j]))
                            cluster_lists.append((datum["FaceB"][j], cpu_z_list[1], datum["Label"][j]))

                        stream_outputs, stream_recons, stream_inputs, mu_lists, std_lists, z_sample_lists, class_predictions = None, None, None, None, None, None, None

                    # Here, cluster the cluster_lists for each cluster
                    for c in range(3):
                        #datas = []
                        cluster_data = []
                        for j in range(len(cluster_lists)):
                            #datas.append((cluster_lists[j][0],cluster_lists[j][2],cluster_lists[j][1][c].numpy())) # image, label, z for specific cluster
                            cluster_data.append(cluster_lists[j][1][c].numpy())

                        kmeans = KMeans(n_clusters=2, random_state=0).fit(cluster_data)

                        for j in range(len(cluster_data)):
                            if kmeans.labels_[j] == 0:
                                opts = {"caption":str(kmeans.labels_[j])+" cluster from z distribution "+str(c)}
                                r = cluster_lists[j][0].clone()
                                r = torch.clamp(r,-1,1)
                                vis.image(self.dbs['train'].ut(r), opts=opts)

                        for j in range(len(cluster_data)):
                            if kmeans.labels_[j] == 1:
                                opts = {"caption":str(kmeans.labels_[j])+" cluster from z distribution "+str(c)}
                                r = cluster_lists[j][0].clone()
                                r = torch.clamp(r,-1,1)
                                vis.image(self.dbs['train'].ut(r), opts=opts)


                    exit()
                else:
                    for i, datum in enumerate(self.dbs['train'].loader):
                        in_datums = [Variable(datum["FaceA"].cuda()), Variable(datum["FaceB"].cuda())]
                        stream_outputs, stream_recons, stream_inputs, mu_lists, std_lists, z_sample_lists, class_predictions = self.VAE(in_datums)

                        if random.randint(0,4) == 4:
                            for s in range(len(in_datums)):
                                vis.image(self.dbs['train'].ut(in_datums[s][0].cpu().data))
                                for c in range(self.VAE.clusters):
                                    vis.image(self.dbs['train'].ut(stream_outputs[s][c][0].cpu().data))

                        loss = self.recon_loss(stream_outputs, stream_inputs, mu_lists, std_lists, class_predictions, Variable(datum["Label"].cuda()))

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
            else:
                # Cluster before saving to see intermediate clustering results for each of the 3 clusters (z exports)

                if self.save_models:
                    self.save_model(epoch=epoch)


            '''
            else:  # Save models and visualization of data

                print("Epoch {}, saving models this time".format(epoch))
                if self.save_models:
                    self.save_model(epoch=epoch)

                # Reconstruction samples
                for i, datum in enumerate(self.db):

                    if i > self.vis_size:
                        break

                    in_datum = Variable(datum["ImageA"].cuda())

                    Z, z_mu, z_var, out_datum = self.VAE(in_datum)

                    inp, outp = in_datum.data, out_datum.data
                    inp, outp = inp.squeeze(0), outp
                    inp, outp = inp.cpu(), outp.cpu()

                    visualize_tensors([inp, outp],file=self.directory_visuals + "/viewpoint_result_e" + str(epoch) + "_b" + str(i).zfill(2))

                # Random z samples
                self.VAE.inference = True

                for i in range(6):
                    sample = Variable(torch.randn(12, self.z_spatial_dim)).cuda()
                    sample = self.VAE.decode(sample)
                    visualize_tensors([sample.data.cpu()],file=self.directory_samples + "/viewpoint_result_e" + str(epoch) + "_b" + str(i).zfill(2))

                self.VAE.inference = False
            '''
    def recon_loss(self, recon_x, x, mu, logvar, class_pred, target):
        L_rec = 0
        L_KLD = 0
        L_cluster = 0
        for stream_idx in range(len(recon_x)):
            for i in range(len(recon_x[stream_idx])):
                stream_in = x[stream_idx]
                stream_out = recon_x[stream_idx][i]
                L_rec += F.mse_loss(stream_out.view(-1, 3 * 224 *224),stream_in.view(-1, 3 * 224* 224))
                L_KLD += -0.5 * torch.sum(1 + logvar[stream_idx][i] - mu[stream_idx][i].pow(2) - logvar[stream_idx][i].exp())

        L_KLD /= recon_x[0][0].size()[0] * 224 * 224 * self.VAE.clusters * len(mu)
        L_KLD = torch.clamp(L_KLD, 0, 1000)
        criterion = nn.CrossEntropyLoss()
        L_class = criterion(class_pred, target.squeeze())
        print("REC",L_rec.data.cpu().numpy(),"KLD", L_KLD.data.cpu().numpy(), "Class", L_class.data.cpu().numpy(), "IntraCluster Entropy", L_cluster)
        return L_rec+L_KLD+L_class

    def loss_function(self, recon_x, x, mu, logvar):
        # Using tanh ([-1,1]) so we need MSE for proper reconstruction
        MSE = F.mse_loss(recon_x.view(-1, 3 * self.sizes[0] * self.sizes[1]),x.view(-1, 3 * self.sizes[0] * self.sizes[1]))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= recon_x.size()[0] * self.sizes[0] * self.sizes[1]
        KLD = torch.clamp(KLD, 0, 1000)  # tiny floating point errors with values like -5e-14

        print(MSE.data.cpu().numpy()[0], KLD.data.cpu().numpy()[0])
        return MSE + KLD

############ TO-DOs
# Scale up to 0.00001 of the training set [x], epoch: 200
# Scale up to 0.01 of the training set [], epoch:

# Include fc prediction loss for relation

########################

def visualize_tensors(tensor_list, file="./sandbox_results/temp", normalize=False):
    # Visualize tensor list (concatenating vertically) and save them in a file
    # accepts a list of tensors of the form batch x channels x (size1 x size2)

    # max channel dimension
    cmax = -1
    proper_size = None

    t_vis = []
    for tensor in tensor_list:
        t_size = tensor.size()[1]
        if cmax < t_size:
            cmax = t_size
        proper_size = tensor.size()

    # expand channels for all tensors with less channels
    for tensor in tensor_list:
        if tensor.size()[1] < cmax:
            t_vis.append(tensor.expand((proper_size[0], cmax, proper_size[2], proper_size[3])))
        else:
            t_vis.append(tensor)

    vis = torch.cat(t_vis, 0)

    if "png" not in file:
        file += ".png"

    # visualize all tensors (greyscale and colored together)
    save(vis, file, nrow=proper_size[0], pad_value=1.0, normalize=normalize)

###########################

################### Model Types

model_types = ["Vanilla","Convolutional","Conditional"]
scheduler_types = {}

encoder_choice, decoder_choice, multipliers_choice,\
batchnorm_choice, z_choice, linear_choice, kernel_choice,\
batch_choice, sizes_choice, window_context_choice, scheduler_choice \
= {},{},{},{},{},{},{},{},{},{},{}

# Scheduler types

scheduler_types['Vanilla A'] = {
            'epochs' : 3000,
            'lr_epoch' : 30000,
            'lr_mod' : 1.0,
            'lr_change' : 0.1,
            'lr_base' : 1e-3,
            'wd' : 0}

scheduler_types['Conditional A'] = {
            'lr_base' : 1e-3,
            'wd' : 0}

# Convolutional model types

model_type = 'FaceConvVae'+' '+model_types[1]
encoder_choice[model_type] = ['C','CM','C','CM','C','CM','CM']
decoder_choice[model_type] = list(reversed(encoder_choice[model_type]))
multipliers_choice[model_type] = [1,1,2,2,4,4,4,4]
batchnorm_choice[model_type] = False
z_choice[model_type] = 32
linear_choice[model_type] = 256
kernel_choice[model_type] = 32
sizes_choice[model_type] = (224,224)
scheduler_choice[model_type] = scheduler_types['Vanilla A']

# Convolutional model types

model_type = 'FaceConvVaeMixture'+' '+model_types[1]
encoder_choice[model_type] = ['C','CM','C','CM','C','CM','CM']
decoder_choice[model_type] = list(reversed(encoder_choice[model_type]))
multipliers_choice[model_type] = [1,1,2,2,4,4,4,4]
batchnorm_choice[model_type] = False
z_choice[model_type] = 32
linear_choice[model_type] = 256
kernel_choice[model_type] = 32
sizes_choice[model_type] = (224,224)
scheduler_choice[model_type] = scheduler_types['Vanilla A']

class FaceConvVAE(nn.Module):

    def cpu_mode_sampling(self, cpu_mode=True):
        self.cpu_mode = cpu_mode

    def sample_z(self, mu, log_var):

        # Using reparameterization trick to sample from a gaussian
        if self.inference:
            return mu
        if self.cpu_mode:
            eps = Variable(torch.randn(mu.size()[0], self.z_spatial_dim))
        else:
            eps = Variable(torch.randn(mu.size()[0], self.z_spatial_dim)).cuda()
        return mu + torch.exp(log_var / 2) * eps

    def make_layers_encoder(self, cfg):

        layers = []
        in_channels = 3

        i = 0

        for v in cfg:
            r = self.kernel_multipliers[i]
            if v == 'CM':
                conv2d = nn.Conv2d(in_channels, r, kernel_size=3, stride=2, padding=1)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(r), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = r
                i += 1
            elif v == 'C':
                conv2d = nn.Conv2d(in_channels, r, kernel_size=3, stride=1, padding=1)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(r), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = r
                i += 1

        return nn.Sequential(*layers)

    def make_layers_decoder(self, cfg):

        layers = []
        in_channels = self.kernel_multipliers[self.last_conv]

        i = 0

        for v in cfg:

            r = self.kernel_multipliers[self.last_conv - i - 1]
            if self.last_conv - i - 1 < 0:
                r = 3

            if v == 'CM':
                conv2d = nn.ConvTranspose2d(in_channels, r, kernel_size=2, stride=2, padding=0, output_padding=0)

                layers += [nn.ReLU(inplace=True), conv2d]
                in_channels = r
                i += 1
            elif v == 'C':
                conv2d = nn.Conv2d(in_channels, r, kernel_size=3, stride=1, padding=1)

                layers += [nn.ReLU(inplace=True), conv2d]
                in_channels = r
                i += 1

        return nn.Sequential(*layers)

    def __init__(self, cfg_encoder=None, cfg_decoder=None, linear_nodes=None, kernel_base=None, z_spatial_dim=None,
                 sizes=None, batch_norm=None, kernel_multipliers=None):
        super(FaceConvVAE, self).__init__()
        self.cpu_mode = False
        self.inference = False
        self.z_spatial_dim = z_spatial_dim
        self.kernel_base = kernel_base
        self.linear_nodes = linear_nodes
        self.sizes = sizes

        self.max_pools = len([d for d in cfg_encoder if d == 'CM'])
        self.kernel_multipliers = [x * self.kernel_base for x in kernel_multipliers]
        self.batch_norm = batch_norm
        self.last_conv = len(cfg_encoder) - 1

        self.tanh_in = nn.Tanh()
        self.encoder_spatial_cnn = self.make_layers_encoder(cfg=cfg_encoder)
        self.encoder_spatial_linear = nn.Sequential(

            nn.Linear(int(self.kernel_multipliers[self.last_conv] * (self.sizes[0] / (2 ** self.max_pools)) ** 2),
                      self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes, self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes, self.linear_nodes),
            nn.ReLU(),
        )

        self.z_mu = nn.Linear(int(self.linear_nodes), z_spatial_dim)  # mean of Z
        self.z_var = nn.Linear(int(self.linear_nodes), z_spatial_dim)  # Log variance s^2 of Z
        self.decode_z = nn.Linear(z_spatial_dim, int(self.linear_nodes))

        self.decoder_spatial_linear = nn.Sequential(

            nn.ReLU(),
            nn.Linear(int(self.linear_nodes), self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes, self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes, int(
                kernel_base * kernel_multipliers[self.last_conv] * (self.sizes[0] / (2 ** self.max_pools)) ** 2)),
        )

        self.decoder_spatial_cnn = self.make_layers_decoder(cfg=cfg_decoder)

        self.tanh_out = nn.Tanh()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, myinput):
        myinput = self.tanh_in(myinput)
        e_l = self.encoder_spatial_cnn(myinput)
        e_l = e_l.view(myinput.size()[0], -1)
        e_l = self.encoder_spatial_linear(e_l)

        z_mu = self.z_mu(e_l)
        z_var = self.z_var(e_l)
        Z_sample = self.sample_z(z_mu, z_var)

        Z = self.decode_z(Z_sample)
        Z = self.decoder_spatial_linear(Z)
        Z = Z.view(myinput.size()[0], self.kernel_multipliers[self.last_conv],
                   int(self.sizes[0] / (2 ** self.max_pools)), int(self.sizes[1] / (2 ** self.max_pools)))
        Z = self.decoder_spatial_cnn(Z)

        output = self.tanh_out(Z)

        return Z_sample, z_mu, z_var, output

    def decode(self, Z_sample):
        Z = self.decode_z(Z_sample)
        Z = self.decoder_spatial_linear(Z)
        Z = Z.view(Z_sample.size()[0], self.kernel_multipliers[self.last_conv],
                   int(self.sizes[0] / (2 ** self.max_pools)), int(self.sizes[1] / (2 ** self.max_pools)))

        Z = self.decoder_spatial_cnn(Z)
        output = self.tanh_out(Z)

        return output

class FaceConvVAEMixture(nn.Module):

    def cpu_mode_sampling(self, cpu_mode=True):
        self.cpu_mode = cpu_mode

    def sample_z(self, mu, log_var):

        # Using reparameterization trick to sample from a gaussian mixture
        if self.inference:
            return mu
        if self.cpu_mode:
            eps = Variable(torch.randn(mu.size()[0], self.z_spatial_dim))
        else:
            eps = Variable(torch.randn(mu.size()[0], self.z_spatial_dim)).cuda()
        return mu + torch.exp(log_var / 2) * eps

    def make_layers_encoder(self, cfg):

        layers = []
        in_channels = 3

        i = 0

        for v in cfg:
            r = self.kernel_multipliers[i]
            if v == 'CM':
                conv2d = nn.Conv2d(in_channels, r, kernel_size=3, stride=2, padding=1)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(r), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = r
                i += 1
            elif v == 'C':
                conv2d = nn.Conv2d(in_channels, r, kernel_size=3, stride=1, padding=1)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(r), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = r
                i += 1

        return nn.Sequential(*layers)

    def make_layers_decoder(self, cfg):

        layers = []
        in_channels = self.kernel_multipliers[self.last_conv]

        i = 0

        for v in cfg:

            r = self.kernel_multipliers[self.last_conv - i - 1]
            if self.last_conv - i - 1 < 0:
                r = 3

            if v == 'CM':
                conv2d = nn.ConvTranspose2d(in_channels, r, kernel_size=2, stride=2, padding=0, output_padding=0)

                layers += [nn.ReLU(inplace=True), conv2d]
                in_channels = r
                i += 1
            elif v == 'C':
                conv2d = nn.Conv2d(in_channels, r, kernel_size=3, stride=1, padding=1)

                layers += [nn.ReLU(inplace=True), conv2d]
                in_channels = r
                i += 1

        return nn.Sequential(*layers)

    def __init__(self, cfg_encoder=None, cfg_decoder=None, linear_nodes=None, kernel_base=None, z_spatial_dim=None,
                 sizes=None, batch_norm=None, kernel_multipliers=None):
        super(FaceConvVAEMixture, self).__init__()
        self.cpu_mode = False
        self.inference = False
        self.z_spatial_dim = z_spatial_dim
        self.kernel_base = kernel_base
        self.linear_nodes = linear_nodes
        self.sizes = sizes
        self.components = 2

        self.max_pools = len([d for d in cfg_encoder if d == 'CM'])
        self.kernel_multipliers = [x * self.kernel_base for x in kernel_multipliers]
        self.batch_norm = batch_norm
        self.last_conv = len(cfg_encoder) - 1

        self.tanh_in = nn.Tanh()
        self.encoder_spatial_cnn = self.make_layers_encoder(cfg=cfg_encoder)
        self.encoder_spatial_linear = nn.Sequential(

            nn.Linear(int(self.kernel_multipliers[self.last_conv] * (self.sizes[0] / (2 ** self.max_pools)) ** 2),
                      self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes, self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes, self.linear_nodes),
            nn.ReLU(),
        )

        #self.z_mu = [None for x in range(self.components)]
        #self.z_var = [None for x in range(self.components)]

        self.z_mu_1 = nn.Linear(int(self.linear_nodes), z_spatial_dim) # mean of Z
        self.z_var_1 = nn.Linear(int(self.linear_nodes), z_spatial_dim)  # Log variance s^2 of Z
        self.z_mu_2 = nn.Linear(int(self.linear_nodes), z_spatial_dim) # mean of Z
        self.z_var_2 = nn.Linear(int(self.linear_nodes), z_spatial_dim)  # Log variance s^2 of Z

        self.decode_z = nn.Linear(z_spatial_dim, int(self.linear_nodes))

        self.decoder_spatial_linear = nn.Sequential(

            nn.ReLU(),
            nn.Linear(int(self.linear_nodes), self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes, self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes, int(
                kernel_base * kernel_multipliers[self.last_conv] * (self.sizes[0] / (2 ** self.max_pools)) ** 2)),
        )

        self.decoder_spatial_cnn = self.make_layers_decoder(cfg=cfg_decoder)

        self.tanh_out = nn.Tanh()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, myinput):
        myinput = self.tanh_in(myinput)
        e_l = self.encoder_spatial_cnn(myinput)
        e_l = e_l.view(myinput.size()[0], -1)
        e_l = self.encoder_spatial_linear(e_l)

        i = random.randint(0, self.components-1)

        if i == 0:
            z_mu = self.z_mu_1(e_l)
            z_var = self.z_var_1(e_l)
        else:
            z_mu = self.z_mu_2(e_l)
            z_var = self.z_var_2(e_l)

        Z_sample = self.sample_z(z_mu, z_var)

        Z = self.decode_z(Z_sample)
        Z = self.decoder_spatial_linear(Z)
        Z = Z.view(myinput.size()[0], self.kernel_multipliers[self.last_conv],
                   int(self.sizes[0] / (2 ** self.max_pools)), int(self.sizes[1] / (2 ** self.max_pools)))
        Z = self.decoder_spatial_cnn(Z)

        output = self.tanh_out(Z)

        return Z_sample, z_mu, z_var, output

    def decode(self, Z_sample):
        Z = self.decode_z(Z_sample)
        Z = self.decoder_spatial_linear(Z)
        Z = Z.view(Z_sample.size()[0], self.kernel_multipliers[self.last_conv],
                   int(self.sizes[0] / (2 ** self.max_pools)), int(self.sizes[1] / (2 ** self.max_pools)))

        Z = self.decoder_spatial_cnn(Z)
        output = self.tanh_out(Z)

        return output

class VAE_Adventures():

    def load_model(self, epoch=1000):
        self.VAE.load_state_dict(torch.load(self.directory_models + "model_" + str(epoch) + ".model"))
        self.inference = True

    def save_model(self, epoch=0):
        torch.save(self.VAE.state_dict(), self.directory_models + "model_" + str(epoch) + ".model")

    def __init__(self, model_version="X", model_choice="X", save_models=True, \
                 model_type="X", resources=None, result_path=None):


        self.save_models = save_models

        self.model_version = model_version
        self.model_type = model_type
        self.model_choice = model_choice + " " + model_type
        self.db = resources

        self.cfg_encoder = encoder_choice[self.model_choice]
        self.cfg_decoder = decoder_choice[self.model_choice]
        self.kernel_base = kernel_choice[self.model_choice]
        self.kernel_multipliers = multipliers_choice[self.model_choice]
        self.batch_norm = batchnorm_choice[self.model_choice]
        self.main_directory = "Convolutional VAE results/"

        self.z_spatial_dim = z_choice[self.model_choice]
        self.linear_nodes = linear_choice[self.model_choice]

        self.sizes = sizes_choice[self.model_choice]

        self.datasplit = (8, 10)  # train on 8 out of 10 batches
        self.vis_size = 8

        self.result_path = result_path
        self.scheduler = scheduler_choice[self.model_choice]

        ### Directories and Logging Definition

        self.root_directory = self.result_path
        self.version_directory = self.model_version + " " + str(self.model_choice)

        self.directory = self.root_directory + self.main_directory + self.version_directory
        print("Using directory : {}".format(self.directory))

        pathlib.Path(self.directory).mkdir(parents=True, exist_ok=True)


        self.directory_models = self.directory + "/models/"
        self.directory_visuals = self.directory + "/visuals/"
        self.directory_visuals_test = self.directory + "/visuals_test/"
        self.directory_samples = self.directory + "/samples/"
        self.directory_samples_varied = self.directory + "/samples_varied/"

        pathlib.Path(self.directory_models).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.directory_visuals).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.directory_samples).mkdir(parents=True, exist_ok=True)

        self.VAE = FaceConvVAEMixture(cfg_encoder=self.cfg_encoder, cfg_decoder=self.cfg_decoder, sizes=self.sizes,
                               z_spatial_dim=self.z_spatial_dim, linear_nodes=self.linear_nodes,
                               kernel_base=self.kernel_base, kernel_multipliers=self.kernel_multipliers,
                               batch_norm=self.batch_norm).cuda()

    def train(self):

        # Training hyperparameters
        self.VAE.inference = False

        lr_epoch = self.scheduler['lr_epoch']
        lr_mod = self.scheduler['lr_mod']
        lr_change = self.scheduler['lr_change']
        lr_base = self.scheduler['lr_base']
        wd = self.scheduler['wd']
        epochs = self.scheduler['epochs']

        samples_vary_creation = False
        starting_epoch = 1
        epoch_save_all = 100

        if starting_epoch != 1:
            directory_models = self.directory + "/models/"
            self.VAE.load_state_dict(torch.load(self.directory_models + "model_" + str(starting_epoch - 1) + ".model"))

        self.optimizer = optim.Adam(self.VAE.parameters(), weight_decay=wd, lr=lr_base * lr_mod)

        # Start training
        for epoch in range(starting_epoch, epochs + 1):

            print("Epoch {}".format(epoch))
            if epoch % lr_epoch == 0:
                lr_mod *= lr_change
                self.optimizer = optim.Adam(self.VAE.parameters(), weight_decay=wd, lr=lr_base * lr_mod)

            if epoch % epoch_save_all != 0:
                for i, datum in enumerate(self.db):
                    print("In batch {} / {}".format(i, len(self.db) / 256))

                    in_datum = Variable(datum["ImageA"].cuda())

                    Z, z_mu, z_var, out_datum = self.VAE(in_datum)

                    loss = self.loss_function(out_datum, in_datum, z_mu, z_var)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()



            else:  # Save models and visualization of data

                print("Epoch {}, saving models this time".format(epoch))
                if self.save_models:
                    self.save_model(epoch=epoch)

                # Reconstruction samples
                for i, datum in enumerate(self.db):

                    if i > self.vis_size:
                        break

                    in_datum = Variable(datum["ImageA"].cuda())

                    Z, z_mu, z_var, out_datum = self.VAE(in_datum)

                    inp, outp = in_datum.data, out_datum.data
                    inp, outp = inp.squeeze(0), outp
                    inp, outp = inp.cpu(), outp.cpu()

                    visualize_tensors([inp, outp],file=self.directory_visuals + "/viewpoint_result_e" + str(epoch) + "_b" + str(i).zfill(2))

                # Random z samples
                self.VAE.inference = True

                for i in range(6):
                    sample = Variable(torch.randn(12, self.z_spatial_dim)).cuda()
                    sample = self.VAE.decode(sample)
                    visualize_tensors([sample.data.cpu()],file=self.directory_samples + "/viewpoint_result_e" + str(epoch) + "_b" + str(i).zfill(2))

                self.VAE.inference = False

    def loss_function(self, recon_x, x, mu, logvar):
        # Using tanh ([-1,1]) so we need MSE for proper reconstruction
        MSE = F.mse_loss(recon_x.view(-1, 3 * self.sizes[0] * self.sizes[1]),
                         x.view(-1, 3 * self.sizes[0] * self.sizes[1]))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= recon_x.size()[0] * self.sizes[0] * self.sizes[1]
        KLD = torch.clamp(KLD, 0, 1000)  # tiny floating point errors with values like -5e-14

        print(MSE.data.cpu().numpy()[0], KLD.data.cpu().numpy()[0])
        return MSE + KLD