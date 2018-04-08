from recombined_gan import *

environments = ['main','plots','kernels','class','class_dsc','rec','sample','cluster','recombine']
vis = {}
for env in environments:
    vis[env] = visdom.Visdom(env=env)
    vis[env].close()

#################### Vanilla Model Double Stream

schedulers_types = {'DCGAN':{'epochs' : 30, 'start':1, 'save':1, 'test':1, 'cluster':1, 'lr': 10e-5, 'wd': 0}}
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

    def __init__(self, save_models=True, model_choice="X", dbs=None, result_path=None, proper_size=(224,224)):

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

        self.optimizer = None

        self.main_directory = "FaceGan Lab/"
        self.result_path = result_path
        self.scheduler = schedulers_types[self.model_choice]

        self.root_directory = self.result_path
        self.version_directory = self.model_choice
        self.directory = GAN_Building.use_path(self.root_directory + self.main_directory + self.version_directory)
        self.directory_models = GAN_Building.use_path(self.directory + "/models/")
        self.directory_visuals = GAN_Building.use_path(self.directory + "/visuals/")

        self.GAN = RecombinedGAN(complexity_type[self.model_choice], proper_size=self.proper_size).cuda()


    def visualize_train(self, in_datums, stream_inputs, stream_outputs, d_labels, d_pred, datum, class_predictions, domain_predictions, epoch):
        for s in range(len(in_datums)//2):
            vis['rec'].image(torch.clamp(self.dbs['train'].ut(stream_inputs[s][0].cpu().data), 0, 1), opts={'caption': "Input to Recreate (e{})".format(epoch)})
            vis['rec'].image(torch.clamp(self.dbs['train'].ut(stream_outputs[s][0].cpu().data), 0, 1), opts={'caption': "Reconstructed Output (e{})".format(epoch)})
            if self.GAN.GAN[1].fake:
                pred = "[]"
                if d_labels[s][0] == d_pred[s][0].max(0)[1].cpu().data.numpy()[0]:
                    pred = "[x]"
                vis['class_dsc'].image(torch.clamp(self.dbs['train'].ut(stream_inputs[s][0].cpu().data), 0, 1), opts={'caption': '{}'.format(pred)})
            else:
                pred = "[]"
                if d_labels[s][0] == d_pred[s][0].max(0)[1].cpu().data.numpy()[0]:
                    pred = "[x]"
                vis['class_dsc'].image(torch.clamp(self.dbs['train'].ut(stream_outputs[s][0].cpu().data), 0, 1), opts={'caption': '{}'.format(pred)})

        for s in range(len(in_datums)):
            pred_16 = "[]"
            pred_5 = "[]"
            if datum["normal"]["Label"][0].numpy()[0] + 1 == 1 + class_predictions[0].max(0)[1].cpu().data.numpy()[0]:
                pred_16 = "[x]"
            if datum["normal"]["Domain"][0].numpy()[0] + 1 == 1 + domain_predictions[0].max(0)[1].cpu().data.numpy()[0]:
                pred_5 = "[x]"
            if s <= 1:
                vis['class'].image(torch.clamp(self.dbs['train'].ut(stream_inputs[s][0].cpu().data), 0, 1), opts={'caption': 't {} vs p {} {} ~ 16, t {} vs p {} {}~ 5'.format(datum["normal"]["Label"][0].numpy()[0] + 1, 1 + class_predictions[0].max(0)[1].cpu().data.numpy()[0],pred_16, datum["normal"]["Domain"][0].numpy()[0] + 1, 1 + domain_predictions[0].max(0)[1].cpu().data.numpy()[0], pred_5)})

    def train(self):

        self.optimizer = optim.Adam(self.GAN.parameters(), weight_decay=self.scheduler['wd'], lr= self.scheduler['lr'])
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
            self.GAN.GAN[0].inference = False
            self.GAN.train()

            the_epoch_visual = True
            with progressbar.ProgressBar(max_value=len(self.dbs['train'].loader)) as bar:
                for batch_idx, datum in enumerate(self.dbs['train'].loader):

                    self.GAN.GAN[1].fake = not self.GAN.GAN[1].fake
                    self.batch_counting['train'].append(batch_idx+(epoch-1)*len(self.dbs['train'].loader))

                    in_datums = [Variable(datum['normal']["FaceA"].cuda()), Variable(datum['normal']["FaceB"].cuda()),Variable(datum['random']["FaceA"].cuda()), Variable(datum['random']["FaceB"].cuda())]
                    stream_outputs, stream_recons, stream_inputs, mu_lists, std_lists, z_sample_lists, class_predictions, domain_predictions, d_pred, d_labels, d_pred_r, d_labels_r = self.GAN(in_datums)

                    if the_epoch_visual:
                        self.visualize_train(in_datums, stream_inputs, stream_outputs, d_labels, d_pred, datum, class_predictions, domain_predictions, epoch)
                        if batch_idx > 2:
                            the_epoch_visual = False

                    L_REC, L_KLD, L_class, L_domain, L_Gen, L_Dsc = self.total_loss(stream_outputs, stream_inputs, mu_lists, std_lists, class_predictions, domain_predictions, Variable(datum["normal"]["Label"].cuda()),  Variable(datum["normal"]["Domain"].cuda()), d_pred, d_labels,d_pred_r, d_labels_r)
                    L_total = sum([L_REC, L_KLD, L_class, L_domain, L_Gen, L_Dsc])

                    if type(L_Gen) == type(1):
                        L_Gen = Variable(torch.zeros(1)).cuda()

                    for grouper_info in range(1):
                        self.batches_plot_loss['train']['loss class'].append(np.double(L_class.cpu().data.numpy()))
                        self.batches_plot_loss['train']['loss domain'].append(np.double(L_domain.cpu().data.numpy()))
                        self.batches_plot_loss['train']['loss kld'].append(np.double(L_KLD.cpu().data.numpy()))
                        self.batches_plot_loss['train']['loss mse'].append(np.double(L_REC.cpu().data.numpy()))
                        self.batches_plot_loss['train']['loss gen'].append(np.double(L_Gen.cpu().data.numpy()))
                        self.batches_plot_loss['train']['loss dsc'].append(np.double(L_Dsc.cpu().data.numpy()))


                        self.epoch_plot_loss['train']['loss class'][epoch-self.scheduler['start']] += (np.double(L_class.cpu().data.numpy()))
                        self.epoch_plot_loss['train']['loss domain'][epoch-self.scheduler['start']] += (np.double(L_domain.cpu().data.numpy()))
                        self.epoch_plot_loss['train']['loss kld'][epoch-self.scheduler['start']] += (np.double(L_KLD.cpu().data.numpy()))
                        self.epoch_plot_loss['train']['loss mse'][epoch-self.scheduler['start']] +=(np.double(L_REC.cpu().data.numpy()))
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
                            predictions = d_pred[stream].max(1)[1].type_as(class_targets)
                            correct_dsc = np.double(predictions.eq(class_targets).data.numpy())
                            correct_discriminator += correct_dsc.sum().sum()
                            current_correct_sum += correct_dsc.sum().sum()

                        proper_total_discriminator_batch = in_datums[0].size()[0]
                        if self.GAN.GAN[1].fake:
                            proper_total_discriminator_batch *= 6
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

                        K = 20
                        if len(self.batch_counting['train']) > K:
                            lenn = len(self.batch_counting['train'])-K
                            self.batch_counting['train'] = self.batch_counting['train'][lenn:]
                            for key in self.batches_plot_loss['train'].keys():
                                self.batches_plot_loss['train'][key] = self.batches_plot_loss['train'][key][lenn:]
                            for key in self.batches_plot_acc['train'].keys():
                                self.batches_plot_acc['train'][key] = self.batches_plot_acc['train'][key][lenn:]




                    self.optimizer.zero_grad()
                    L_total.backward(retain_graph=True)
                    #self.check_gradients()
                    self.optimizer.step()
                    bar.update(batch_idx)

            for grouper_visualizations in range(1):
                self.epoch_plot_acc['train']['acc class'].append(correct_class*100.0/total_batch)
                trace = dict(x=self.epoch_counting['train'], y=self.epoch_plot_acc['train']['acc class'], mode="markers+lines", type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                layout = dict(title="Classification Accuracy Plot (train)", xaxis={'title': 'epoch idx'}, yaxis={'title': 'class accuracy'})
                vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Class accuracy, training (epoch)'})

                self.epoch_plot_acc['train']['acc dsc'].append(correct_discriminator * 100/ (4*total_batch))
                trace = dict(x=self.epoch_counting['train'], y=self.epoch_plot_acc['train']['acc dsc'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                layout = dict(title="Discriminator Accuracy Plot (train)", xaxis={'title': 'epoch idx'},yaxis={'title': 'class accuracy'})
                vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Discriminator accuracy, training (epoch)'})

                trace1 = dict(x=self.epoch_counting['train'], y=self.epoch_plot_loss['train']['loss dsc'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"}, name='disc')
                trace2 = dict(x=self.epoch_counting['train'], y=self.epoch_plot_loss['train']['loss gen'], mode="markers+lines",type='custom', marker={'color': 'blue', 'symbol': 0, 'size': "5"}, name='gen')
                trace3 = dict(x=self.epoch_counting['train'], y=self.epoch_plot_loss['train']['loss class'], mode="markers+lines",type='custom', marker={'color': 'cyan', 'symbol': 0, 'size': "5"}, name='class')
                trace4 = dict(x=self.epoch_counting['train'], y=self.epoch_plot_loss['train']['loss mse'], mode="markers+lines",type='custom', marker={'color': 'purple', 'symbol': 0, 'size': "5"}, name='mse')
                trace5 = dict(x=self.epoch_counting['train'], y=self.epoch_plot_loss['train']['loss kld'], mode="markers+lines",type='custom', marker={'color': 'black', 'symbol': 0, 'size': "5"}, name='kld')
                layout = dict(title="Losses Plot (train)", xaxis={'title': 'epoch idx'}, yaxis={'title': 'All losses'})
                vis['plots']._send({'data': [trace1, trace2, trace3, trace4, trace5], 'layout': layout, 'win': 'All losses, training (epoch)'})
            for grouper_z_samples in range(1):
                # Create samples for Z from last z vector in training
                self.GAN.GAN[0].eval()
                sample_output, _, _ = self.GAN.GAN[0].decode(z_sample_lists[0])
                vis['sample'].image(torch.clamp(self.dbs['train'].ut(sample_output[0].cpu().data),0,1),  opts={'caption':"Sample Rec"})
                vis['sample'].image(torch.clamp(torch.add(torch.mul(torch.cat(z_sample_lists[0],1)[0].unsqueeze(0).unsqueeze(0).repeat(1,24,1).cpu().data,0.1), 0.5),0,1),  opts={'caption':"Sample Rec vector",'width':self.proper_size[0],'height':self.proper_size[1]})

                # Vary samples of z
                for i in range(self.GAN.GAN[0].clusters):
                    z_random_full = []
                    for j in range(len(z_sample_lists[0])):
                        z_random_full.append(z_sample_lists[0][j].clone())
                    randomized_z = Variable(torch.normal(torch.zeros((self.GAN.GAN[0].z)),torch.ones(((self.GAN.GAN[0].z)))), volatile=True).cuda()
                    z_random_full[i] = torch.add(z_random_full[i], randomized_z)
                    sample_output, _, _ = self.GAN.GAN[0].decode(z_random_full)
                    vis['sample'].image(torch.clamp(self.dbs['train'].ut(sample_output[0].cpu().data),0,1),  opts={'caption':"Sample Randomized {}".format(i+1)})
                    vis['sample'].image(torch.clamp(torch.add(torch.mul(torch.cat(z_random_full,1)[0].unsqueeze(0).unsqueeze(0).repeat(1,24,1).cpu().data,0.1), 0.5),0,1),  opts={'caption':"Sample Randomized vector {}".format(i+1),'width':self.proper_size[0],'height':self.proper_size[1]})

            continue

            if epoch % self.scheduler['test'] == 0:
                epoch_test = int(math.floor(epoch/self.scheduler['test']))
                self.epoch_counting['test'].append(epoch_test)
                print("TESTING -----------------")
                self.GAN.GAN[0].inference = True
                self.GAN.GAN[0].eval()
                self.GAN.GAN[1].eval()

                bar = progressbar.ProgressBar()
                with progressbar.ProgressBar(max_value=len(self.dbs['test'].loader)) as bar:

                    correct_class = 0
                    correct_domain = 0
                    correct_discriminator = 0
                    total_batch = 0

                    self.epoch_plot_loss['test']['loss class'].append(0)
                    self.epoch_plot_loss['test']['loss domain'].append(0)
                    self.epoch_plot_loss['test']['loss kld'].append(0)
                    self.epoch_plot_loss['test']['loss mse'].append(0)
                    self.epoch_plot_loss['test']['loss gen'].append(0)
                    self.epoch_plot_loss['test']['loss dsc'].append(0)

                    the_epoch_visual = True
                    for i, datum in enumerate(self.dbs['test'].loader):
                        self.GAN.GAN[1].fake = not self.GAN.GAN[1].fake

                        in_datums = [Variable(datum['normal']["FaceA"].cuda()), Variable(datum['normal']["FaceB"].cuda()), Variable(datum['random']["FaceA"].cuda()), Variable(datum['random']["FaceB"].cuda())]

                        stream_outputs, stream_recons, stream_inputs, mu_lists, std_lists, z_sample_lists, class_predictions, domain_predictions, d_pred, d_labels, d_pred_r, d_labels_r = self.GAN(in_datums)

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
                            predictions = d_pred[stream].max(1)[1].type_as(class_targets)
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

                    self.epoch_plot_acc['test']['acc dsc'].append(correct_discriminator * 100 / (2 * total_batch))
                    trace = dict(x=self.epoch_counting['test'], y=self.epoch_plot_acc['test']['acc dsc'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                    layout = dict(title="Discriminator Accuracy Plot (test)", xaxis={'title': 'epoch idx'},yaxis={'title': 'class accuracy'})
                    vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Discriminator accuracy, testing (epoch)'})

                    trace1 = dict(x=self.epoch_counting['test'], y=self.epoch_plot_loss['test']['loss dsc'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"}, name='disc')
                    trace2 = dict(x=self.epoch_counting['test'], y=self.epoch_plot_loss['test']['loss gen'], mode="markers+lines", type='custom', marker={'color': 'blue', 'symbol': 0, 'size': "5"}, name='gen')
                    trace3 = dict(x=self.epoch_counting['test'], y=self.epoch_plot_loss['test']['loss class'], mode="markers+lines", type='custom', marker={'color': 'cyan', 'symbol': 0, 'size': "5"}, name='class')
                    trace3b = dict(x=self.epoch_counting['test'], y=self.epoch_plot_loss['test']['loss domain'], mode="markers+lines", type='custom', marker={'color': 'orange', 'symbol': 0, 'size': "5"}, name='domain')
                    trace4 = dict(x=self.epoch_counting['test'], y=self.epoch_plot_loss['test']['loss mse'], mode="markers+lines",type='custom', marker={'color': 'purple', 'symbol': 0, 'size': "5"}, name='mse')
                    trace5 = dict(x=self.epoch_counting['test'], y=self.epoch_plot_loss['test']['loss kld'], mode="markers+lines",type='custom', marker={'color': 'black', 'symbol': 0, 'size': "5"}, name='kld')
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
                    stream_outputs, stream_recons, stream_inputs, mu_lists, std_lists, z_sample_lists, class_predictions, domain_predictions, d_pred, d_labels, d_pred_r, d_labels_r = self.GAN(in_datums)

                    n_clusters = 2
                    for j in range(self.GAN.GAN[0].clusters):
                        print("Clustering cluster {}".format(j))
                        cluster_list_z = []
                        cluster_list_orig = []
                        for i in range(stream_inputs[0].size()[0]):
                            cluster_list_z.append(z_sample_lists[0][j][i].cpu().data.numpy())
                            cluster_list_orig.append(stream_inputs[0][i].cpu().data)

                        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(cluster_list_z)
                        for clusters_kmeans in range(n_clusters):
                            for i in range(stream_inputs[0].size()[0]):
                                if kmeans.labels_[i] == clusters_kmeans:
                                    vis['cluster'].image(torch.clamp(self.dbs['train'].ut(cluster_list_orig[i]),0,1), opts={'caption':"latent group {}, cluster: {}".format(j, kmeans.labels_[i])})

                    break

    def total_loss(self, recon_x, x, mu, logvar, class_pred, domain_pred, target, target_domain, d_pred, d_labels, d_pred_r, d_labels_r):

        streams = len(recon_x)
        proper_streams = streams//2
        L_REC, L_REC_scale = 0, 1
        L_KLD, L_KLD_scale = 0, 1
        L_class, L_class_scale = 0, 1
        L_domain, L_domain_scale = 0, 1
        L_Gen, L_Gen_scale = 0, 1
        L_Dsc, L_Dsc_scale = 0, 1

        batch_size = recon_x[0].size()[0]
        pss = self.proper_size[0] * self.proper_size[1] # proper size squared

        for stream_idx in range(streams):
            stream_in = x[stream_idx]
            stream_out = recon_x[stream_idx]
            mu_cat = torch.cat(mu[stream_idx],dim=1)
            logvar_cat = torch.cat(logvar[stream_idx],dim=1)
            L_REC += F.mse_loss(stream_out.view(-1, 1 * pss),stream_in.view(-1, 1 * pss))
            try:
                L_KLD += -0.5 * torch.sum(1 + logvar_cat - mu_cat.pow(2) - logvar_cat.exp())
            except:
                L_KLD += -0.5 * torch.sum(1 - mu_cat.pow(2))

        L_KLD /= batch_size * pss * len(recon_x)
        L_KLD = torch.clamp(L_KLD, 0, 1000)

        criterion = nn.CrossEntropyLoss()
        L_class = criterion(class_pred, target.squeeze())
        L_domain = criterion(domain_pred, target_domain.squeeze())

        for stream in range(streams): # the randomized streams for real examples
            # real targets are 1s, fake targets are 0s / from code for discriminator
            # We want the generator to get a small loss only if the predictions match real targets (1s)
            # since that means that we succeeded in fooling the discriminator (it predicted real labels from our reconstructions)
            discriminator_labels = Variable(torch.ones(batch_size).long()).cuda()
            if self.GAN.GAN[1].fake:
                L_Gen += criterion(d_pred[stream], discriminator_labels)

            L_Dsc += criterion(d_pred[stream], Variable(d_labels[stream].cuda()))

        # Add a loss for the recombined samples only during fake phases
        for stream in range(proper_streams):
            if self.GAN.GAN[1].fake:
                discriminator_labels = [Variable(torch.ones(batch_size).long()).cuda() for i in range(proper_streams)]
                L_Gen += criterion(d_pred_r[stream], discriminator_labels[stream])
                L_Dsc += criterion(d_pred_r[stream], Variable(d_labels_r[stream].cuda()))

        L_REC*= L_REC_scale
        L_KLD*= L_KLD_scale
        L_class*= L_class_scale
        L_domain*= L_domain_scale
        L_Gen*= L_Gen_scale
        L_Dsc*= L_Dsc_scale
        return L_REC, L_KLD, L_class, L_domain, L_Gen, L_Dsc

########################
