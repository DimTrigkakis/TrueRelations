import torch
import torch.nn as nn

from torch.autograd import Variable
import pathlib

import torch.optim as optim
import math
import random
import torch.nn.functional as F

from torchvision.utils import save_image as save

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