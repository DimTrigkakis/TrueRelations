from recombined_gan import *

environments = ['main','plots','kernels','class','class_dsc','rec','sample','cluster','recombine']
vis = {}
for env in environments:
    vis[env] = visdom.Visdom(env=env)
    vis[env].close()

#################### Vanilla Model Double Stream

schedulers_types = {'DCGAN':{'epochs' : 10, 'start':1, 'save':1, 'test':1, 'cluster':1, 'lr': 2e-4, 'wd': 0}}
complexity_type = {'DCGAN': 'Complexity A'}

model_choice = 'DCGAN' # Build Scheduler for training and Parameters for model architecture

class GAN_Building():

    def load_model(self, epoch=1000):
        self.GAN.load_state_dict(torch.load(self.directory_models + "GAN_model_" + str(epoch) + ".model"))
        self.GAN.GAN[0].inference = False

    def save_model(self, epoch=0):
        torch.save(self.GAN.state_dict(), self.directory_models + "GAN_model_" + str(epoch) + ".model")

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
        self.epoch_counting = {'train':[], 'test':[]}
        self.epoch_plot_acc = {'train':{'acc class':[],'acc domain':[],'acc dsc':[]}, 'test':{'acc class':[], 'acc domain':[],'acc dsc':[]}}
        self.epoch_plot_loss = {'train':{'loss class':[], 'loss domain':[],'loss dsc':[], 'loss mse':[],'loss kld':[],'loss gen':[]}, 'test':{'loss class':[], 'loss domain':[],'loss dsc':[], 'loss mse':[],'loss kld':[],'loss gen':[]}}

        self.batch_counting = {'train':[]}
        self.batches_plot_loss = {'train':{'loss class':[], 'loss domain':[],'loss dsc':[], 'loss mse':[],'loss kld':[],'loss gen':[]}}
        self.batches_plot_acc = {'train':{'acc class':[], 'acc domain':[],'acc dsc':[]}}

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

        #
        dataparallel = False
        if dataparallel:
            torch.cuda.set_device(device=2)
            self.GAN_generator_dp = nn.DataParallel(FaceVAEMixture(complexity=complexity_type[self.model_choice], proper_size=self.proper_size, channels=3), device_ids=[2,3]).cuda()
            self.GAN_discriminator_dp = nn.DataParallel(FaceDiscriminator(complexity=complexity_type[self.model_choice], channels=3, image_size=self.proper_size[0]), device_ids=[2,3]).cuda()
            self.GAN_generator = self.GAN_generator_dp.module
            self.GAN_discriminator = self.GAN_discriminator_dp.module
        else:
            self.GAN_generator = FaceVAEMixture(complexity=complexity_type[self.model_choice], proper_size=self.proper_size, channels=3).cuda()
            self.GAN_encoder = self.GAN_generator.encoder
            self.GAN_decoder = self.GAN_generator.decoder
            self.GAN_discriminator = FaceDiscriminator(complexity=complexity_type[self.model_choice], channels=3, image_size=self.proper_size[0]).cuda()

            # The following are very important
            self.GAN_decoder.weight_init(0,0.02)
            self.GAN_encoder.weight_init(0,0.02)
            self.GAN_discriminator.discriminate.weight_init(0,0.02)

    def visualize_stream(self, stream_sample, category='sample', title=''):
        vis[category].image(torch.clamp(self.dbs['train'].ut(stream_sample.cpu().data), 0, 1), opts={'caption': title, 'width': 64, 'height': 64})

    def update_info(self, title, info, op=None):
        # input: title, comma separated groups like "train,batch,A loss"
        #      : info, cuda Variable containing a single value, or a python number / string
        #      : op tuple, available options are 'augment', which given a position, updates the value by adding the info to the position e.g. ('augment',0)
        if type(info) == Variable:
            info = np.double(info.cpu().data.numpy())

        if op is None:
            if title not in self.plot_info.keys():
                self.plot_info[title] = []
            self.plot_info[title].append(info)
        elif op[0] == 'augment':
            if title not in self.plot_info.keys():
                self.plot_info[title] = [0]
            if op[1] == len(self.plot_info[title]):
                self.plot_info[title].append[0]
            assert op[1] < len(self.plot_info[title])
            self.plot_info[title][op[1]] += info

    def hook_activations(self, active):
        self.activations[self.x_deep_type][self.x_stream] = active
        return
    def train(self):

        self.optimizer_encoder = optim.Adam(self.GAN_encoder.parameters(), lr=self.scheduler['lr'], betas=(0.5, 0.999))
        self.optimizer_decoder = optim.Adam(self.GAN_decoder.parameters(), lr=self.scheduler['lr'], betas=(0.5, 0.999))
        self.optimizer_discriminator = optim.Adam(self.GAN_discriminator.parameters(), lr=self.scheduler['lr'], betas=(0.5, 0.999))

        if self.scheduler['start'] != 1:
            self.directory_models = self.directory + "/models/"
            self.GAN.load_state_dict(torch.load(self.directory_models + "GAN_model_" + str(self.scheduler['start'] - 1) + ".model"))

        for epoch in range(self.scheduler['start'], self.scheduler['epochs'] + 1):

            for grouper_info in range(1):
                self.epoch_counting['train'].append(epoch)
                for key in self.epoch_plot_loss['train'].keys():
                    self.epoch_plot_loss['train'][key].append(0)

                correct_class = 0
                correct_domain = 0
                correct_discriminator = 0
                total_batch = 0

            print("Training commences ----------------- epoch {}".format(epoch))
            self.GAN_generator.inference = False
            self.GAN_discriminator.train()
            self.GAN_generator.train()

            the_epoch_visual = True

            self.plot_info = {}

            with progressbar.ProgressBar(max_value=len(self.dbs['train'].loader)) as bar:
                for batch_idx, datum in enumerate(self.dbs['train'].loader):
                    self.update_info("train,batch,count", batch_idx+(epoch-1)*len(self.dbs['train'].loader))
                    in_datums = [Variable(datum["Face"].cuda())]

                    '''
                    self.activations = [[None for i in range(len(in_datums))] for j in range(2)]
                    self.x_deep_type = 0
                    self.x_stream = 0
                    '''

                    ##################################

                    ######################################################
                    # Encoder phase

                    self.optimizer_encoder.zero_grad()

                    mvars = [self.GAN_generator.encoder(in_datums[stream]) for stream in range(len(in_datums))]
                    stream_reconstructions = [None for stream in range(len(in_datums))]
                    for stream in range(len(in_datums)):
                        mvars[stream][0], mvars[stream][1] = torch.cat(mvars[stream][0], 1), torch.cat(mvars[stream][1], 1)
                        stream_sample = self.GAN_generator.sample_z(mvars[stream])
                        stream_reconstructions[stream] = (self.GAN_generator.decoder(stream_sample))

                    VAE_mse, VAE_kld = self.vae_loss(in_datums, stream_reconstructions, mvars)

                    '''
                    final_relu = self.GAN_discriminator.discriminate.discriminator[10]
                    final_relu.register_forward_hook(lambda m, i, o: self.hook_activations(o))

                    for stream in range(len(in_datums)):
                        self.x_stream = stream
                        self.x_deep_type = 0
                        self.GAN_discriminator([in_datums[stream]])

                    for stream in range(len(in_datums)):
                        self.x_stream = stream
                        self.x_deep_type = 1
                        self.GAN_discriminator([stream_reconstructions[stream]])

                    loss_deep_mse, loss_deep_mse_multiplier = 0, 1
                    for stream in range(len(in_datums)):
                        loss_deep_mse += self.gaussian_loss(self.activations[0][stream], self.activations[1][stream])
                    loss_deep_mse = torch.mul(loss_deep_mse, loss_deep_mse_multiplier)
                    '''

                    VAE_total_loss =VAE_mse+VAE_kld

                    VAE_total_loss.backward(retain_graph=True)
                    self.optimizer_encoder.step()
                    
                    self.update_info("train,batch,mse loss", VAE_mse)
                    self.update_info("train,batch,kld loss", VAE_kld)
                    self.update_info("train,epoch,mse loss", VAE_mse, op=('augment', epoch-self.scheduler['start']))
                    self.update_info("train,epoch,kld loss", VAE_kld, op=('augment', epoch-self.scheduler['start']))

                    # Generator phase on reconstruction and gan objective, to train decoder

                    self.optimizer_decoder.zero_grad()
                    #mvars_gen = [self.GAN_generator.encoder(in_datums[stream]) for stream in range(len(in_datums))]
                    #stream_reconstructions_gen = [None for stream in range(len(in_datums))]
                    #for stream in range(len(in_datums)):
                    #    mvars_gen[stream][0], mvars_gen[stream][1] = torch.cat(mvars_gen[stream][0], 1), torch.cat(mvars_gen[stream][1], 1)
                    #    stream_sample_gen = self.GAN_generator.sample_z(mvars_gen[stream])
                    #    stream_reconstructions_gen[stream] = (self.GAN_generator.decoder(stream_sample_gen))

                    L_dec_mse, _ = self.vae_loss(in_datums, stream_reconstructions, mvars)
                    gamma = 0.01
                    L_dec_mse = torch.mul(L_dec_mse, gamma)

                    # trained on Dis(Dec(Enc(x)))
                    discriminator_prediction_streams_gen = self.GAN_discriminator([stream_reconstructions[stream] for stream in range(len(in_datums))])
                    true_label_streams_gen = [torch.ones(len(in_datums[0]), 1) for stream in range(len(in_datums))]

                    L_Gen_fool = self.discriminator_loss(discriminator_prediction_streams_gen, true_label_streams_gen)

                    # trained on Dis(Dec(p(z)))
                    noise = [Variable(torch.randn((len(in_datums[0]), self.GAN_generator.z)).view(-1, self.GAN_generator.z, 1, 1)).cuda()]

                    discriminator_prediction_streams_noise = self.GAN_discriminator([self.GAN_generator.decoder(noise[stream]) for stream in range(len(in_datums))])
                    true_label_streams = [torch.ones(len(in_datums[0]), 1) for stream in range(len(in_datums))]
                    L_Gen_noise_fool = self.discriminator_loss(discriminator_prediction_streams_noise, true_label_streams)

                    L_enc_total = L_Gen_fool + L_Gen_noise_fool + L_dec_mse
                    L_enc_total.backward(retain_graph=True)

                    self.optimizer_decoder.step()

                    ######################### Discriminator update
                    # Real phase for discriminator
                    self.optimizer_discriminator.zero_grad()

                    discriminator_prediction_streams_real = self.GAN_discriminator(in_datums)
                    true_label_streams = [torch.ones(len(in_datums[0]), 1) for stream in range(len(in_datums))]
                    L_Dsc_real = self.discriminator_loss(discriminator_prediction_streams_real, true_label_streams)

                    # Fake phase for discriminator with random VAE noise
                    #noise = [Variable(torch.randn((len(in_datums[0]), self.GAN_generator.z)).view(-1, self.GAN_generator.z, 1, 1)).cuda()]
                    #discriminator_prediction_streams = self.GAN_discriminator([self.GAN_generator.decoder(noise[stream]) for stream in range(len(in_datums))])
                    fake_label_streams = [torch.zeros(len(in_datums[0]), 1) for stream in range(len(in_datums))]
                    L_Dsc_fake = self.discriminator_loss(discriminator_prediction_streams_noise, fake_label_streams)

                    # Fake phase for discriminator with reconstructed VAE images
                    #mvars_gen = [self.GAN_generator.encoder(in_datums[stream]) for stream in range(len(in_datums))]
                    #stream_reconstructions_gen = [None for stream in range(len(in_datums))]
                    #for stream in range(len(in_datums)):
                    #    mvars_gen[stream][0], mvars_gen[stream][1] = torch.cat(mvars_gen[stream][0], 1), torch.cat(mvars_gen[stream][1], 1)
                    #    stream_sample_gen = self.GAN_generator.sample_z(mvars_gen[stream])
                    #    stream_reconstructions_gen[stream] = (self.GAN_generator.decoder(stream_sample_gen))

                    #discriminator_prediction_streams_gen = self.GAN_discriminator([stream_reconstructions[stream] for stream in range(len(in_datums))])
                    true_label_streams_gen = [torch.zeros(len(in_datums[0]), 1) for stream in range(len(in_datums))]
                    L_Dsc_fake_rec = self.discriminator_loss(discriminator_prediction_streams_gen, true_label_streams_gen)

                    L_Gan_Dsc = L_Dsc_real + L_Dsc_fake + L_Dsc_fake_rec
                    L_Gan_Dsc.backward()

                    self.optimizer_discriminator.step()

                    # Visualization update
                    if random.random() > 0.0:
                        self.visualize_stream(in_datums[0][0], 'rec', 'input image')
                        self.visualize_stream(self.GAN_generator.decoder(noise[0])[0], 'rec', 'random noise rec')
                        self.visualize_stream(stream_reconstructions[0][0], 'rec', 'sample rec')

                    #trace_loss_mse = dict(x=self.plot_info['train,batch,count'], y=self.plot_info['train,batch,mse loss'], mode="markers+lines", type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"}, name='mse')
                    #trace_loss_kld = dict(x=self.plot_info['train,batch,count'], y=self.plot_info['train,batch,kld loss'], mode="markers+lines", type='custom', marker={'color': 'black', 'symbol': 0, 'size': "5"}, name='kld')

                    #layout = dict(title="Losses Plot (train)", xaxis={'title': 'batch idx'}, yaxis={'title': 'All train batch losses'})
                    #vis['plots']._send({'data': [trace_loss_mse, trace_loss_kld], 'layout': layout, 'win': 'All losses, train, batch'})

                    bar.update(batch_idx)
                    continue

                    for grouper_info in range(1):
                        self.batches_plot_loss['train']['loss class'].append(np.double(L_class.cpu().data.numpy()))
                        self.batches_plot_loss['train']['loss domain'].append(np.double(L_domain.cpu().data.numpy()))
                        self.batches_plot_loss['train']['loss gen'].append(np.double(L_Gen.cpu().data.numpy()))
                        self.batches_plot_loss['train']['loss dsc'].append(np.double(L_Dsc.cpu().data.numpy()))

                        self.epoch_plot_loss['train']['loss class'][epoch-self.scheduler['start']] += (np.double(L_class.cpu().data.numpy()))
                        self.epoch_plot_loss['train']['loss domain'][epoch-self.scheduler['start']] += (np.double(L_domain.cpu().data.numpy()))
                        self.epoch_plot_loss['train']['loss gen'][epoch-self.scheduler['start']] +=(np.double(L_Gen.cpu().data.numpy()))
                        self.epoch_plot_loss['train']['loss dsc'][epoch-self.scheduler['start']] +=(np.double(L_Dsc.cpu().data.numpy()))

                        class_targets = Variable(datum["normal"]["Label"])
                        domain_targets = Variable(datum["normal"]["Domain"])
                        predictions = class_predictions.max(1)[1].cpu()
                        predictions_domain = domain_predictions.max(1)[1].cpu()
                        correct = np.double(predictions.eq(class_targets.squeeze(1)).data.numpy())
                        correct_d = np.double(predictions_domain.eq(domain_targets.squeeze(1)).data.numpy())
                        correct_class += (correct.sum())
                        correct_domain += (correct_d.sum())
                        total_batch += in_datums[0].size()[0]
                        self.batches_plot_acc['train']['acc class'].append(100*np.double(correct.sum())/ in_datums[0].size()[0])
                        self.batches_plot_acc['train']['acc domain'].append(100*np.double(correct_d.sum())/ in_datums[0].size()[0])

                        current_correct_sum = 0
                        for stream in range(len(stream_outputs)):
                            class_targets = Variable(d_labels[stream])
                            predictions = d_pred[stream].type_as(class_targets)
                            conditional_predictions = np.where(predictions.data.cpu().numpy() < 0.5, 0 , 1)
                            correct_dsc = np.double(np.equal(conditional_predictions,class_targets.data.cpu().numpy()))
                            correct_discriminator += correct_dsc.sum().sum()
                            current_correct_sum += correct_dsc.sum().sum()


                        proper_total_discriminator_batch = in_datums[0].size()[0]
                        if self.GAN.GAN[1].fake:
                            proper_total_discriminator_batch *= 4
                        else:
                            proper_total_discriminator_batch *= 4

                        self.batches_plot_acc['train']['acc dsc'].append(current_correct_sum * 100 / proper_total_discriminator_batch)


                        ### Plot visualizations
                        trace = dict(x=self.batch_counting['train'], y=self.batches_plot_acc['train']['acc class'], mode="markers+lines", type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                        layout = dict(title="Classification Accuracy Plot (train)", xaxis={'title': 'batch idx'}, yaxis={'title': 'class accuracy'})
                        vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Class accuracy, training'})

                        trace = dict(x=self.batch_counting['train'], y=self.batches_plot_acc['train']['acc domain'], mode="markers+lines", type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                        layout = dict(title="Domain Accuracy Plot (train)", xaxis={'title': 'batch idx'}, yaxis={'title': 'domain accuracy'})
                        vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Domain accuracy, training'})

                        trace = dict(x=self.batch_counting['train'], y=self.batches_plot_acc['train']['acc dsc'], mode="markers+lines", type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                        layout = dict(title="Discriminator Accuracy Plot (train)", xaxis={'title': 'batch idx'}, yaxis={'title': 'discriminator accuracy'})
                        vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Discriminator accuracy, training'})

                        # fix running averages for generator and discriminator
                        trace1_pre = list(self.batches_plot_loss['train']['loss dsc'])
                        trace1_plot = [i for i in trace1_pre]
                        for i in range(len(trace1_pre) - 1):
                            trace1_plot[i] = (trace1_pre[i] + trace1_pre[i + 1]) / 2

                        trace2_pre = list(self.batches_plot_loss['train']['loss gen'])
                        trace2_plot = [i for i in trace2_pre]
                        for i in range(len(trace2_pre) - 1):
                            trace2_plot[i] = (trace2_pre[i] + trace2_pre[i + 1]) / 2

                        trace1 = dict(x=self.batch_counting['train'], y=trace1_plot, mode="markers+lines", type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"}, name='disc')
                        trace2 = dict(x=self.batch_counting['train'], y=trace2_plot, mode="markers+lines", type='custom', marker={'color': 'blue', 'symbol': 0, 'size': "5"}, name='gen')
                        trace3 = dict(x=self.batch_counting['train'], y=self.batches_plot_loss['train']['loss class'], mode="markers+lines", type='custom', marker={'color': 'cyan', 'symbol': 0, 'size': "5"}, name='class')
                        trace3_b = dict(x=self.batch_counting['train'], y=self.batches_plot_loss['train']['loss domain'], mode="markers+lines", type='custom', marker={'color': 'orange', 'symbol': 0, 'size': "5"}, name='domain')
                        trace4 = dict(x=self.batch_counting['train'], y=self.batches_plot_loss['train']['loss mse'], mode="markers+lines", type='custom', marker={'color': 'purple', 'symbol': 0, 'size': "5"}, name='mse')
                        trace5 = dict(x=self.batch_counting['train'], y=self.batches_plot_loss['train']['loss kld'], mode="markers+lines", type='custom', marker={'color': 'black', 'symbol': 0, 'size': "5"}, name='kld')
                        layout = dict(title="Losses Plot (train)", xaxis={'title': 'batch idx'}, yaxis={'title': 'All losses'})
                        vis['plots']._send({'data': [trace1, trace2, trace3, trace3_b, trace4, trace5], 'layout': layout, 'win': 'All losses, training'})

                        K = 200
                        if len(self.batch_counting['train']) > K:
                            lenn = len(self.batch_counting['train'])-K
                            self.batch_counting['train'] = self.batch_counting['train'][lenn:]
                            for key in self.batches_plot_loss['train'].keys():
                                self.batches_plot_loss['train'][key] = self.batches_plot_loss['train'][key][lenn:]
                            for key in self.batches_plot_acc['train'].keys():
                                self.batches_plot_acc['train'][key] = self.batches_plot_acc['train'][key][lenn:]

            continue

            for grouper_visualizations in range(1):
                self.epoch_plot_acc['train']['acc class'].append(correct_class*100.0/total_batch)
                trace = dict(x=self.epoch_counting['train'], y=self.epoch_plot_acc['train']['acc class'], mode="markers+lines", type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                layout = dict(title="Classification Accuracy Plot (train)", xaxis={'title': 'epoch idx'}, yaxis={'title': 'class accuracy'})
                vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Class accuracy, training (epoch)'})

                self.epoch_plot_acc['train']['acc domain'].append(correct_domain * 100.0 / total_batch)
                trace = dict(x=self.epoch_counting['train'], y=self.epoch_plot_acc['train']['acc domain'], mode="markers+lines", type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                layout = dict(title="Domain Accuracy Plot (train)", xaxis={'title': 'epoch idx'}, yaxis={'title': 'domain accuracy'})
                vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Domain accuracy, training (epoch)'})

                self.epoch_plot_acc['train']['acc dsc'].append(correct_discriminator * 100/ (4*total_batch))
                trace = dict(x=self.epoch_counting['train'], y=self.epoch_plot_acc['train']['acc dsc'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                layout = dict(title="Discriminator Accuracy Plot (train)", xaxis={'title': 'epoch idx'},yaxis={'title': 'class accuracy'})
                vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Discriminator accuracy, training (epoch)'})

                trace1 = dict(x=self.epoch_counting['train'], y=[i/total_batch for i in self.epoch_plot_loss['train']['loss dsc']], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"}, name='disc')
                trace2 = dict(x=self.epoch_counting['train'], y=[i/total_batch for i in self.epoch_plot_loss['train']['loss gen']], mode="markers+lines",type='custom', marker={'color': 'blue', 'symbol': 0, 'size': "5"}, name='gen')
                trace3 = dict(x=self.epoch_counting['train'], y=[i/total_batch for i in self.epoch_plot_loss['train']['loss class']], mode="markers+lines",type='custom', marker={'color': 'cyan', 'symbol': 0, 'size': "5"}, name='class')
                trace3_b = dict(x=self.epoch_counting['train'], y=[i/total_batch for i in self.epoch_plot_loss['train']['loss domain']], mode="markers+lines", type='custom', marker={'color': 'orange', 'symbol': 0, 'size': "5"}, name='domain')

                trace4 = dict(x=self.epoch_counting['train'], y=[i/total_batch for i in self.epoch_plot_loss['train']['loss mse']], mode="markers+lines",type='custom', marker={'color': 'purple', 'symbol': 0, 'size': "5"}, name='mse')
                trace5 = dict(x=self.epoch_counting['train'], y=[i/total_batch for i in self.epoch_plot_loss['train']['loss kld']], mode="markers+lines",type='custom', marker={'color': 'black', 'symbol': 0, 'size': "5"}, name='kld')
                layout = dict(title="Losses Plot (train)", xaxis={'title': 'epoch idx'}, yaxis={'title': 'All losses'})
                vis['plots']._send({'data': [trace1, trace2, trace3, trace3_b, trace4, trace5], 'layout': layout, 'win': 'All losses, training (epoch)'})
            for grouper_z_samples in range(1):
                # Create samples for Z from last z vector in training
                self.GAN.GAN[0].eval()
                last_z_sample = torch.cat(z_sample_lists[0], 1)
                sample_output = self.GAN.GAN[0].decoder(last_z_sample)
                vis['sample'].image(torch.clamp(self.dbs['train'].ut(sample_output[0].cpu().data),0,1),  opts={'caption':"Sample Rec"})
                vis['sample'].image(torch.clamp(torch.add(torch.mul(last_z_sample[0].repeat(1,24,1).cpu().data,0.1), 0.5),0,1).permute(2,1,0),  opts={'caption':"Sample Rec vector",'width':self.proper_size[0],'height':self.proper_size[1]})

                # Vary samples of z
                for i in range(self.GAN.GAN[0].clusters):
                    z_random_full = []
                    for j in range(len(z_sample_lists[0])):
                        z_random_full.append(z_sample_lists[0][j].clone())
                    randomized_z = Variable(torch.normal(torch.zeros((self.GAN.GAN[0].z//self.GAN.GAN[0].clusters,1,1)),torch.ones(((self.GAN.GAN[0].z//self.GAN.GAN[0].clusters,1,1)))), volatile=True).cuda()

                    z_random_full[i] = torch.add(z_random_full[i], randomized_z)
                    sample_output = self.GAN.GAN[0].decoder(torch.cat(z_random_full,1))
                    vis['sample'].image(torch.clamp(self.dbs['train'].ut(sample_output[0].cpu().data),0,1),  opts={'caption':"Sample Randomized {}".format(i+1)})
                    vis['sample'].image(torch.clamp(torch.add(torch.mul(torch.cat(z_random_full,1)[0].repeat(1,24,1).cpu().data,0.1), 0.5),0,1).permute(2,1,0),  opts={'caption':"Sample Randomized vector {}".format(i+1),'width':self.proper_size[0],'height':self.proper_size[1]})
            if epoch % self.scheduler['test'] == 0:
                epoch_test = int(math.floor(epoch/self.scheduler['test']))

                for grouper_info in range(1):
                    self.epoch_counting['test'].append(epoch_test)
                    for key in self.epoch_plot_loss['test'].keys():
                        self.epoch_plot_loss['test'][key].append(0)

                    correct_class = 0
                    correct_domain = 0
                    correct_discriminator = 0
                    total_batch = 0

                print("TESTING -----------------")
                self.GAN.GAN[0].inference = True
                self.GAN.eval()

                bar = progressbar.ProgressBar()
                with progressbar.ProgressBar(max_value=len(self.dbs['val'].loader)) as bar:


                    the_epoch_visual = True
                    for i, datum in enumerate(self.dbs['val'].loader):

                        in_datums = [Variable(datum['normal']["FaceA"].cuda()), Variable(datum['normal']["FaceB"].cuda()), Variable(datum['random']["FaceA"].cuda()), Variable(datum['random']["FaceB"].cuda())]

                        stream_outputs, stream_inputs, mu_lists, std_lists, z_sample_lists, class_predictions, domain_predictions, d_pred, d_labels, d_pred_r, d_labels_r = self.GAN(in_datums)

                        if the_epoch_visual:
                            for s in range(len(in_datums)):
                                vis['rec'].image(torch.clamp(self.dbs['test'].ut(stream_inputs[s][0].cpu().data),0,1), opts={'caption':"Test Original"})
                                vis['rec'].image(torch.clamp(self.dbs['test'].ut(stream_outputs[s][0].cpu().data),0,1),  opts={'caption':"Test Reconstruction (overfitting litmus test)"})
                                the_epoch_visual = False

                        L_REC, L_KLD, L_class, L_domain, L_Gen, L_Dsc = self.total_loss(stream_outputs, stream_inputs,mu_lists, std_lists, class_predictions, domain_predictions,Variable(datum["normal"]["Label"].cuda(), volatile=True) ,Variable(datum["normal"]["Domain"].cuda(), volatile=True),d_pred, d_labels, d_pred_r, d_labels_r)

                        if type(L_Gen) == type(1):
                            L_Gen = Variable(torch.zeros(1)).cuda()

                        self.epoch_plot_loss['test']['loss class'][epoch_test-self.scheduler['start']] += (np.double(L_class.cpu().data.numpy()))
                        self.epoch_plot_loss['test']['loss domain'][epoch_test-self.scheduler['start']] += (np.double(L_domain.cpu().data.numpy()))
                        self.epoch_plot_loss['test']['loss kld'][epoch_test-self.scheduler['start']] += (np.double(L_KLD.cpu().data.numpy()))
                        self.epoch_plot_loss['test']['loss mse'][epoch_test-self.scheduler['start']] += (np.double(L_REC.cpu().data.numpy()))
                        self.epoch_plot_loss['test']['loss gen'][epoch_test-self.scheduler['start']] += (np.double(L_Gen.cpu().data.numpy()))
                        self.epoch_plot_loss['test']['loss dsc'][epoch_test-self.scheduler['start']] += (np.double(L_Dsc.cpu().data.numpy()))

                        class_targets = Variable(datum["normal"]["Label"], volatile=True)
                        domain_targets = Variable(datum["normal"]["Domain"], volatile=True)
                        predictions = class_predictions.max(1)[1].cpu()
                        predictions_domain = domain_predictions.max(1)[1].cpu()
                        correct = np.double(predictions.eq(class_targets.squeeze(1)).data.numpy())
                        correct_d = np.double(predictions_domain.eq(domain_targets.squeeze(1)).data.numpy())
                        correct_class += (correct.sum())
                        correct_domain += (correct_d.sum())
                        total_batch += in_datums[0].size()[0]
                        self.epoch_plot_acc['test']['acc class'].append(100 * np.double(correct.sum()) / in_datums[0].size()[0])
                        self.epoch_plot_acc['test']['acc domain'].append(100 * np.double(correct_d.sum()) / in_datums[0].size()[0])

                        for stream in range(len(stream_outputs)):
                            class_targets = Variable(d_labels[stream])
                            predictions = d_pred[stream].type_as(class_targets)
                            conditional_predictions = np.where(predictions.data.cpu().numpy() < 0.5, 0 , 1)
                            correct_dsc = np.double(np.equal(conditional_predictions,class_targets.data.cpu().numpy()))
                            correct_dsc = np.double(predictions.eq(class_targets).data.numpy())
                            correct_discriminator += correct_dsc.sum()

                        bar.update(i)

                # Visualizations
                for grouper_visualizations in range(1):
                    self.epoch_plot_acc['test']['acc class'].append(correct_class * 100.0 / total_batch)
                    trace = dict(x=self.epoch_counting['test'], y=self.epoch_plot_acc['test']['acc class'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                    layout = dict(title="Classification Accuracy Plot (test)", xaxis={'title': 'epoch idx'},yaxis={'title': 'class accuracy'})
                    vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Class accuracy, testing (epoch)'})

                    trace = dict(x=self.epoch_counting['test'], y=self.epoch_plot_acc['test']['acc domain'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                    layout = dict(title="Domain Accuracy Plot (test)", xaxis={'title': 'epoch idx'},yaxis={'title': 'domain accuracy'})
                    vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Domain accuracy, testing (epoch)'})

                    self.epoch_plot_acc['test']['acc dsc'].append(correct_discriminator * 100 / (4* total_batch))
                    trace = dict(x=self.epoch_counting['test'], y=self.epoch_plot_acc['test']['acc dsc'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                    layout = dict(title="Discriminator Accuracy Plot (test)", xaxis={'title': 'epoch idx'},yaxis={'title': 'class accuracy'})
                    vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Discriminator accuracy, testing (epoch)'})

                    trace1 = dict(x=self.epoch_counting['test'], y=[i/total_batch for i in self.epoch_plot_loss['test']['loss dsc']], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"}, name='disc')
                    trace2 = dict(x=self.epoch_counting['test'], y=[i/total_batch for i in self.epoch_plot_loss['test']['loss gen']], mode="markers+lines", type='custom', marker={'color': 'blue', 'symbol': 0, 'size': "5"}, name='gen')
                    trace3 = dict(x=self.epoch_counting['test'], y=[i/total_batch for i in self.epoch_plot_loss['test']['loss class']], mode="markers+lines", type='custom', marker={'color': 'cyan', 'symbol': 0, 'size': "5"}, name='class')
                    trace3b = dict(x=self.epoch_counting['test'], y=[i/total_batch for i in self.epoch_plot_loss['test']['loss domain']], mode="markers+lines", type='custom', marker={'color': 'orange', 'symbol': 0, 'size': "5"}, name='domain')
                    trace4 = dict(x=self.epoch_counting['test'], y=[i/total_batch for i in self.epoch_plot_loss['test']['loss mse']], mode="markers+lines",type='custom', marker={'color': 'purple', 'symbol': 0, 'size': "5"}, name='mse')
                    trace5 = dict(x=self.epoch_counting['test'], y=[i/total_batch for i in self.epoch_plot_loss['test']['loss kld']], mode="markers+lines",type='custom', marker={'color': 'black', 'symbol': 0, 'size': "5"}, name='kld')
                    layout = dict(title="Losses Plot (test)", xaxis={'title': 'epoch idx'}, yaxis={'title': 'All losses'})
                    vis['plots']._send({'data': [trace1, trace2, trace3, trace3b, trace4, trace5], 'layout': layout,'win': 'All losses, testing (epoch)'})
            if epoch % self.scheduler['save'] == 0 and epoch != self.scheduler['start']:
                print("Saving model for epoch:{}".format(epoch))
                if self.save_models:
                    self.save_model(epoch=epoch)
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

