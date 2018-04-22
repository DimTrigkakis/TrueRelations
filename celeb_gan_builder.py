from recombined_gan import *

environments = ['main','plots','kernels','class','class_dsc','rec','sample','cluster','recombine']
vis = {}
for env in environments:
    vis[env] = visdom.Visdom(env=env)
    vis[env].close()

#################### Vanilla Model Double Stream

schedulers_types = {'DCGAN':{'epochs' : 10, 'start':1, 'save':1, 'test':100, 'cluster':100, 'lr': 3e-4, 'wd': 0}}
complexity_type = {'DCGAN': 'Complexity A'}

model_choice = 'DCGAN' # Build Scheduler for training and Parameters for model architecture


class PlotInfo():

    def __init__(self, name):
        self.name = name
        self.plot_info = [] # batch and epoch

    def update(self, value, proper_epoch=None):
        if proper_epoch is not None:
            self.update_info(value, op=('add', proper_epoch))
        else:
            self.update_info(value)

    def update_info(self, info, op=None):
        if type(info) == Variable:
            info = np.double(info.cpu().data.numpy())

        if op is None:
            self.plot_info.append(info)

        elif op[0] == 'add':
            if op[1] == len(self.plot_info):
                self.plot_info.append(0)
            assert op[1] < len(self.plot_info)
            self.plot_info[op[1]] += info

class GAN_Building():

    def save_model(self, epoch=0):
        torch.save(self.GAN_encoder.state_dict(), self.directory_models + "GAN_enc_" + str(epoch) + ".model")
        torch.save(self.GAN_decoder.state_dict(), self.directory_models + "GAN_dec_" + str(epoch) + ".model")
        torch.save(self.GAN_discriminator.state_dict(), self.directory_models + "GAN_dsc_" + str(epoch) + ".model")

    @staticmethod
    def use_path(path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        return path

    def check_gradients(self):

        my_dict = self.GAN.state_dict(keep_vars=True)
        print("")
        for param in my_dict.keys():
            if 'weight' in param or 'bias' in param:
                my_max = torch.max(my_dict[param].grad).cpu().data.numpy()[0]
                my_min = torch.min(torch.abs(my_dict[param].grad)).cpu().data.numpy()[0]
                if my_max > 1e3 or my_min < 1e-20:
                    print("Exploding or vanishing gradient in: {}, min {:0.2f}, max {:0.2f}".format(param, my_min, my_max))

    def __init__(self, save_models=True, model_choice="X", dbs=None, result_path=None, proper_size=(64,64)):

        # Visuals
        self.plot_info = {'batch count':None, 'epoch count':None}
        self.losses = {'vae mse':0, 'vae kld':0, 'dec mse':0, 'dec rec':0, 'dec noise':0, 'dsc real':0, 'dsc fake':0, 'dsc rec':0}
        for loss in self.losses.keys():
            self.plot_info['train,batch,'+loss] = PlotInfo(name='train,batch,'+loss)
            self.plot_info['train,epoch,'+loss] = PlotInfo(name='train,epoch,'+loss)

        self.plot_info['batch count'] = PlotInfo('training batch idx')
        self.plot_info['epoch count'] = PlotInfo('training epoch idx')

        # Experiment parameters
        self.proper_size = proper_size
        self.save_models = save_models
        self.model_choice = model_choice
        self.dbs = dbs
        self.result_path = result_path

        self.main_directory = "FaceGan Lab/"
        self.result_path = result_path
        self.scheduler = schedulers_types[self.model_choice]

        self.root_directory = self.result_path
        self.version_directory = self.model_choice
        self.directory = GAN_Building.use_path(self.root_directory + self.main_directory + self.version_directory)
        self.directory_models = GAN_Building.use_path(self.directory + "/models/")
        self.directory_visuals = GAN_Building.use_path(self.directory + "/visuals/")

        # Model definitions
        self.GAN_generator = FaceVAEMixture(complexity=complexity_type[self.model_choice], proper_size=self.proper_size, channels=3).cuda()
        self.GAN_encoder = self.GAN_generator.encoder
        self.GAN_decoder = self.GAN_generator.decoder
        self.GAN_discriminator = FaceDiscriminator(complexity=complexity_type[self.model_choice], channels=3, image_size=self.proper_size[0]).cuda()

        self.GAN_decoder.weight_init(0,0.02)
        self.GAN_encoder.weight_init(0,0.02)
        self.GAN_discriminator.discriminate.weight_init(0,0.02)

        self.optimizer_encoder = optim.Adam(self.GAN_encoder.parameters(), lr=self.scheduler['lr'], betas=(0.5, 0.999))
        self.optimizer_decoder = optim.Adam(self.GAN_decoder.parameters(), lr=self.scheduler['lr'], betas=(0.5, 0.999))
        self.optimizer_discriminator = optim.Adam(self.GAN_discriminator.parameters(), lr=self.scheduler['lr'], betas=(0.5, 0.999))

    def visualize_stream(self, stream_sample, category='sample', title=''):
        vis[category].image(torch.clamp(self.dbs['train'].ut(stream_sample.cpu().data), 0, 1), opts={'caption': title, 'width': 64, 'height': 64})

    def train(self):

        if self.scheduler['start'] != 1:
            self.directory_models = self.directory + "/models/"
            self.GAN_encoder.load_state_dict(torch.load(self.directory_models + "GAN_enc_" + str(self.scheduler['start'] - 1) + ".model"))
            self.GAN_decoder.load_state_dict(torch.load(self.directory_models + "GAN_dec_" + str(self.scheduler['start'] - 1) + ".model"))
            self.GAN_discriminator.load_state_dict(torch.load(self.directory_models + "GAN_dsc_" + str(self.scheduler['start'] - 1) + ".model"))

        for epoch in range(self.scheduler['start'], self.scheduler['epochs'] + 1):

            for grouper_info in range(1):
                total_batch = 0

            print("Training commences ----------------- epoch {}".format(epoch))
            self.GAN_generator.inference = True
            self.GAN_discriminator.train()
            self.GAN_generator.train()
            self.GAN_decoder.train()

            self.plot_info['epoch count'].update(epoch)

            with progressbar.ProgressBar(max_value=len(self.dbs['train'].loader)) as bar:
                for batch_idx, datum in enumerate(self.dbs['train'].loader):

                    in_datums = [Variable(datum["Face"].cuda())]

                    ######################################################
                    self.plot_info['batch count'].update(batch_idx+(epoch-1)*len(self.dbs['train'].loader))

                    # Encoder phase

                    mvars = [self.GAN_generator.encoder(in_datums[stream]) for stream in range(len(in_datums))]
                    stream_reconstructions = [None for stream in range(len(in_datums))]
                    for stream in range(len(in_datums)):
                        mvars[stream][0], mvars[stream][1] = torch.cat(mvars[stream][0], 1), torch.cat(mvars[stream][1], 1)
                        stream_sample = self.GAN_generator.sample_z(mvars[stream])
                        stream_reconstructions[stream] = (self.GAN_generator.decoder(stream_sample))

                    self.losses['vae mse'], self.losses['vae kld'] = self.vae_loss(in_datums, stream_reconstructions, mvars)
                    VAE_total_loss =self.losses['vae mse']+self.losses['vae kld']

                    self.optimizer_encoder.zero_grad()
                    VAE_total_loss.backward(retain_graph=True)
                    self.optimizer_encoder.step()

                    # Generator phase on reconstruction and gan objective, to train decoder

                    L_dec_mse, _ = self.vae_loss(in_datums, stream_reconstructions, mvars)
                    gamma = 0.01
                    self.losses['dec mse'] = torch.mul(L_dec_mse, gamma)

                    # trained on Dis(Dec(Enc(x)))
                    discriminator_prediction_streams_gen = self.GAN_discriminator([stream_reconstructions[stream] for stream in range(len(in_datums))])
                    true_label_streams_gen = [torch.ones(len(in_datums[0]), 1) for stream in range(len(in_datums))]

                    # trained on Dis(Dec(p(z)))
                    noise = [Variable(torch.randn((len(in_datums[0]), self.GAN_generator.z)).view(-1, self.GAN_generator.z, 1, 1)).cuda()]
                    discriminator_prediction_streams_noise = self.GAN_discriminator([self.GAN_generator.decoder(noise[stream]) for stream in range(len(in_datums))])
                    true_label_streams = [torch.ones(len(in_datums[0]), 1) for stream in range(len(in_datums))]

                    self.losses['dec rec'] = self.discriminator_loss(discriminator_prediction_streams_gen, true_label_streams_gen)
                    self.losses['dec noise'] = self.discriminator_loss(discriminator_prediction_streams_noise, true_label_streams)
                    L_decoder_total = self.losses['dec mse'] + self.losses['dec rec'] + self.losses['dec noise']

                    self.optimizer_decoder.zero_grad()
                    L_decoder_total.backward(retain_graph=True)
                    self.optimizer_decoder.step()

                    ######################### Discriminator update
                    # Real phase for discriminator

                    discriminator_prediction_streams_real = self.GAN_discriminator(in_datums)
                    true_label_streams = [torch.ones(len(in_datums[0]), 1) for stream in range(len(in_datums))]
                    fake_label_streams = [torch.zeros(len(in_datums[0]), 1) for stream in range(len(in_datums))]
                    true_label_streams_gen = [torch.zeros(len(in_datums[0]), 1) for stream in range(len(in_datums))]

                    self.losses['dsc real'] = self.discriminator_loss(discriminator_prediction_streams_real, true_label_streams)
                    self.losses['dsc fake'] = self.discriminator_loss(discriminator_prediction_streams_noise, fake_label_streams)
                    self.losses['dsc rec'] = self.discriminator_loss(discriminator_prediction_streams_gen, true_label_streams_gen)
                    L_Gan_Dsc = self.losses['dsc real'] + self.losses['dsc fake'] + self.losses['dsc rec']

                    self.optimizer_discriminator.zero_grad()
                    L_Gan_Dsc.backward()
                    self.optimizer_discriminator.step()

                    #################### Visualizations
                    for loss in self.losses.keys():
                        self.plot_info['train,batch,'+loss].update(self.losses[loss])
                        self.plot_info['train,epoch,'+loss].update(self.losses[loss], epoch-1)

                    if random.random() > 0.99:
                        self.visualize_stream(in_datums[0][0], 'rec', 'input image')
                        self.visualize_stream(self.GAN_generator.decoder(noise[0])[0], 'rec', 'random noise rec')
                        self.visualize_stream(stream_reconstructions[0][0], 'rec', 'sample rec')

                    total_batch += 1*(in_datums[0].size()[0]/128)
                    plot_traces = []
                    colors = ['red','blue','yellow','green','black','cyan','purple','magenta','brown','pink']
                    color_pick = 0

                    K = 5
                    if len(self.plot_info['batch count'].plot_info) > K:
                        lenn = len(self.plot_info['batch count'].plot_info) - K
                        self.plot_info['batch count'].plot_info = self.plot_info['batch count'].plot_info[lenn:]
                        for key in self.plot_info.keys():
                            if len(str(key).split(',')) == 3 and str(key).split(',')[2] in self.losses.keys() and 'train,batch,' in str(key):
                                self.plot_info[key].plot_info = self.plot_info[key].plot_info[lenn:]

                    for plot in self.plot_info.keys():
                        if len(str(plot).split(',')) == 3 and str(plot).split(',')[2] in self.losses.keys() and 'train,batch,' in str(plot):
                            name = str(plot).split(',')[2]
                            plot_traces.append(dict(x=self.plot_info['batch count'].plot_info, y=self.plot_info[plot].plot_info, mode="markers+lines", type='custom', marker={'color': colors[color_pick], 'symbol': 0, 'size': "5"}, name=name))
                            color_pick += 1

                    layout = dict(title="Batch Loss Plot (train)", xaxis={'title': 'batch idx'}, yaxis={'title': 'loss'})
                    vis['plots']._send({'data': plot_traces, 'layout': layout, 'win': 'All losses, train, batch'})

                    bar.update(batch_idx)

            if epoch % self.scheduler['save'] == 0 and epoch != self.scheduler['start']:
                print("Saving model for epoch:{}".format(epoch))
                if self.save_models:
                    self.save_model(epoch=epoch)

            plot_traces_epochs = []
            color_pick = 0
            for plot in self.plot_info.keys():
                if len(str(plot).split(',')) == 3 and str(plot).split(',')[2] in self.losses.keys() and 'train,epoch,' in str(plot):
                    name = str(plot).split(',')[2]
                    plot_traces_epochs.append(dict(x=self.plot_info['epoch count'].plot_info, y=[i/total_batch for i in self.plot_info[plot].plot_info], mode="markers+lines", type='custom', marker={'color': colors[color_pick], 'symbol': 0, 'size': "5"}, name=name))
                    color_pick += 1

            layout = dict(title="Epoch Loss Plot (train)", xaxis={'title': 'epoch idx'}, yaxis={'title': 'loss'})
            vis['plots']._send({'data': plot_traces_epochs, 'layout': layout, 'win': 'All losses, train, epoch'})

            '''
            if epoch % self.scheduler['cluster'] == 0:
                print("Clustering a batch of train and test samples")

                for i, datum in enumerate(self.dbs['train'].loader):
                    # obtain latent z's in inference mode
                    self.GAN.GAN[0].inference = True
                    self.GAN.GAN[0].eval()
                    in_datums = [Variable(datum['normal']["FaceA"].cuda()), Variable(datum['normal']["FaceB"].cuda()), Variable(datum['random']["FaceA"].cuda()), Variable(datum['random']["FaceB"].cuda())]
                    stream_outputs, stream_inputs, mu_lists, std_lists, z_sample_lists, class_predictions, domain_predictions, d_pred, d_labels, d_pred_r, d_labels_r = self.GAN(in_datums)

                    n_clusters = 2
                    for j in range(self.GAN.GAN[0].clusters):
                        print("Clustering cluster {}".format(j))
                        cluster_list_z = []
                        cluster_list_orig = []
                        for i in range(stream_inputs[0].size()[0]):
                            cluster_list_z.append(z_sample_lists[0][j][i].squeeze().squeeze().cpu().data.numpy())
                            cluster_list_orig.append(stream_inputs[0][i].squeeze().squeeze().cpu().data)

                        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(cluster_list_z)
                        for clusters_kmeans in range(n_clusters):
                            for i in range(stream_inputs[0].size()[0]):
                                if kmeans.labels_[i] == clusters_kmeans:
                                    vis['cluster'].image(torch.clamp(self.dbs['train'].ut(cluster_list_orig[i]),0,1), opts={'caption':"latent group {}, cluster: {}".format(j, kmeans.labels_[i])})

                    break
            '''

    def gaussian_loss(self, x_deep_features, x_rec_deep_features):
        x_deep_features_view = x_deep_features.view(-1)
        x_rec_deep_features_view = x_rec_deep_features.view(-1)
        gloss = torch.mul(torch.sum((x_deep_features_view - x_rec_deep_features_view)**2), 1.0/(128*16*1024))
        return gloss

    def vae_loss(self, in_datums, stream_reconstructions, mvars):

        L_KLD, L_KLD_scale = 0, 1
        L_MSE, L_MSE_scale = 0, 1

        MSE_criterion = nn.MSELoss().cuda()

        for stream in range(len(in_datums)):
            KLD_element = mvars[stream][0].pow(2).add_(mvars[stream][1].exp()).mul_(-1).add_(1).add_(mvars[stream][1])
            L_KLD += torch.sum(KLD_element).mul_(-0.5) / (self.proper_size[0]*self.proper_size[1]*in_datums[0].size()[0])
            L_MSE += MSE_criterion(stream_reconstructions[stream], in_datums[stream])

        L_MSE = torch.mul(L_MSE, L_MSE_scale)
        L_KLD = torch.mul(L_KLD, L_KLD_scale)

        return L_MSE, L_KLD

    def discriminator_loss(self, discriminator_prediction_streams, label_streams):

        L_Dsc, L_Dsc_scale = 0, 1
        sigmoid_criterion = nn.BCELoss().cuda()
        for stream in range(len(label_streams)):
            L_Dsc += sigmoid_criterion(discriminator_prediction_streams[stream], Variable(label_streams[stream].cuda()))
        L_Dsc = torch.mul(L_Dsc, L_Dsc_scale)
        return L_Dsc

########################

