from recombined_gan import *
from sklearn.cluster import KMeans

import pathlib
from sklearn.cluster import KMeans
import progressbar
import resnet_models
import torch.optim as optim
import math
import random
import visdom

import numpy as np
import torch.nn.functional as F

environments = ['main','plots','kernels','class','class_dsc','rec','sample','cluster','recombine']
vis = {}
for env in environments:
    vis[env] = visdom.Visdom(env=env)
    vis[env].close()

#################### Vanilla Model Double Stream

schedulers_types = {'GAN Zodiac':{'epochs' : 1000,'lr_epoch' : 100000, 'start':1,'save':100,'test':25,'cluster':1000,'lr':10e-4}}
complexity_type = {'GAN Zodiac': 'Complexity A'}

model_choice = 'GAN Zodiac' # Build Scheduler for training and Parameters for model architecture

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

        self.proper_size = proper_size
        self.save_models = save_models
        self.model_choice = model_choice
        self.dbs = dbs
        self.result_path = result_path

        self.optimizer = None

        self.main_directory = "MNIST VAE Lab/"
        self.result_path = result_path
        self.scheduler = schedulers_types[self.model_choice]

        self.root_directory = self.result_path
        self.version_directory = self.model_choice
        self.directory = GAN_Building.use_path(self.root_directory + self.main_directory + self.version_directory)
        self.directory_models = GAN_Building.use_path(self.directory + "/models/")
        self.directory_visuals = GAN_Building.use_path(self.directory + "/visuals/")

        self.GAN = RecombinedGAN(complexity_type[self.model_choice], proper_size=self.proper_size).cuda()

    def visualize_train(self, in_datums, stream_inputs, stream_outputs, d_labels, d_pred, datum, class_predictions):
        for s in range(len(in_datums)):
            vis['rec'].image(torch.clamp(self.dbs['train'].ut(stream_inputs[s][0].cpu().data), 0, 1), opts={'caption': "Train Rec"})
            vis['rec'].image(torch.clamp(self.dbs['train'].ut(stream_outputs[s][0].cpu().data), 0, 1), opts={'caption': "Train Rec"})
            if self.GAN.GAN[1].fake:
                vis['class_dsc'].image(torch.clamp(self.dbs['train'].ut(stream_inputs[s][0].cpu().data), 0, 1), opts={'caption': 'target {}, pred {}'.format(d_labels[s][0], d_pred[s][0].max(0)[1].cpu().data.numpy()[0])})
            else:
                vis['class_dsc'].image(torch.clamp(self.dbs['train'].ut(stream_outputs[s][0].cpu().data), 0, 1), opts={'caption': 'target {}, pred {}'.format(d_labels[s][0], d_pred[s][0].max(0)[1].cpu().data.numpy()[0])})

        for s in range(len(in_datums)):
            vis['class'].image(torch.clamp(self.dbs['train'].ut(stream_inputs[s][0].cpu().data), 0, 1), opts={'caption': 'target {}/2, pred {}/2'.format(datum["Diff Labels"][0] + 1, 1 + class_predictions[0].max(0)[1].cpu().data.numpy()[0])})

    def train(self):

        # Start training , testing and visualizing
        epoch_counting = {'train':[], 'test':[]}
        epoch_plot_acc = {'train':{'acc class':[],'acc dsc':[]}, 'test':{'acc class':[],'acc dsc':[]}}
        epoch_plot_loss = {'train':{'loss class':[],'loss dsc':[], 'loss mse':[],'loss kld':[],'loss gen':[]}, 'test':{'loss class':[],'loss dsc':[], 'loss mse':[],'loss kld':[],'loss gen':[]}}

        batch_counting = {'train':[]}
        batches_plot_loss = {'train':{'loss class':[],'loss dsc':[], 'loss mse':[],'loss kld':[],'loss gen':[]}}
        batches_plot_acc = {'train':{'acc class':[],'acc dsc':[]}}

        self.optimizer = optim.Adam(self.GAN.parameters(), weight_decay=0, lr= self.scheduler['lr'])
        if self.scheduler['start'] != 1:
            self.directory_models = self.directory + "/models/"
            self.GAN.load_state_dict(torch.load(self.directory_models + "GAN_model_" + str(self.scheduler['start'] - 1) + ".model"))

        for epoch in range(self.scheduler['start'], self.scheduler['epochs'] + 1):

            for grouper_info in range(1):
                epoch_counting['train'].append(epoch)
                for key in epoch_plot_loss['train'].keys():
                    epoch_plot_loss['train'][key].append(0)
                correct_class = 0
                correct_discriminator = 0
                total_batch = 0

            print("Training commences ----------------- epoch {}".format(epoch))
            self.GAN.GAN[0].inference = False
            self.GAN.train()

            with progressbar.ProgressBar(max_value=len(self.dbs['train'])) as bar:
                for batch_idx, datum in enumerate(self.dbs['train']):
                    self.GAN.GAN[1].fake = not self.GAN.GAN[1].fake

                    batch_counting['train'].append(batch_idx+(epoch-1)*len(self.dbs['train']))
                    in_datums = [Variable(datum["Image Pairs"][0].cuda()), Variable(datum["Image Pairs"][1].cuda())]
                    stream_outputs, stream_recons, stream_inputs, mu_lists, std_lists, z_sample_lists, class_predictions, d_pred, d_labels, d_pred_r, d_labels_r = self.GAN(in_datums)
                    self.visualize_train(in_datums, stream_inputs, stream_outputs, d_labels, d_pred, datum, class_predictions)

                    L_REC, L_KLD, L_class, L_Gen, L_Dsc = self.total_loss(stream_outputs, stream_inputs, mu_lists, std_lists, class_predictions, Variable(datum["Diff Labels"].cuda()), d_pred, d_labels,d_pred_r, d_labels_r)
                    L_total = sum([L_REC, L_KLD, L_class, L_Gen, L_Dsc])

                    if type(L_Gen) == type(1):
                        L_Gen = Variable(torch.zeros(1)).cuda()

                    for grouper_info in range(1):
                        batches_plot_loss['train']['loss class'].append(np.double(L_class.cpu().data.numpy()))
                        batches_plot_loss['train']['loss kld'].append(np.double(L_KLD.cpu().data.numpy()))
                        batches_plot_loss['train']['loss mse'].append(np.double(L_REC.cpu().data.numpy()))
                        batches_plot_loss['train']['loss gen'].append(np.double(L_Gen.cpu().data.numpy()))
                        batches_plot_loss['train']['loss dsc'].append(np.double(L_Dsc.cpu().data.numpy()))

                        epoch_plot_loss['train']['loss class'][epoch-self.scheduler['start']] += (np.double(L_class.cpu().data.numpy()))
                        epoch_plot_loss['train']['loss kld'][epoch-self.scheduler['start']] += (np.double(L_KLD.cpu().data.numpy()))
                        epoch_plot_loss['train']['loss mse'][epoch-self.scheduler['start']] +=(np.double(L_REC.cpu().data.numpy()))
                        epoch_plot_loss['train']['loss gen'][epoch-self.scheduler['start']] +=(np.double(L_Gen.cpu().data.numpy()))
                        epoch_plot_loss['train']['loss dsc'][epoch-self.scheduler['start']] +=(np.double(L_Dsc.cpu().data.numpy()))

                        class_targets = Variable(datum["Diff Labels"])
                        predictions = class_predictions.max(1)[1].type_as(class_targets)
                        correct = np.double(predictions.eq(class_targets).data.numpy())
                        correct_class += (correct.sum())
                        total_batch += in_datums[0].size()[0]
                        batches_plot_acc['train']['acc class'].append(100*np.double(correct.sum())/ in_datums[0].size()[0])

                        current_correct_sum = 0
                        for stream in range(len(stream_outputs)):
                            class_targets = Variable(d_labels[stream])
                            predictions = d_pred[stream].max(1)[1].type_as(class_targets)
                            correct = np.double(predictions.eq(class_targets).data.numpy())
                            correct_discriminator += correct.sum()
                            current_correct_sum += correct.sum()

                        batches_plot_acc['train']['acc dsc'].append(100*current_correct_sum/ (2*(in_datums[0].size()[0])))

                        ### Plot visualizations
                        trace = dict(x=batch_counting['train'], y=batches_plot_acc['train']['acc class'], mode="markers+lines", type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                        layout = dict(title="Classification Accuracy Plot (train)", xaxis={'title': 'batch idx'}, yaxis={'title': 'class accuracy'})
                        vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Class accuracy, training'})

                        trace = dict(x=batch_counting['train'], y=batches_plot_acc['train']['acc dsc'], mode="markers+lines", type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                        layout = dict(title="Discriminator Accuracy Plot (train)", xaxis={'title': 'batch idx'}, yaxis={'title': 'discriminator accuracy'})
                        vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Discriminator accuracy, training'})

                        # fix running averages for generator and discriminator
                        trace1_pre = batches_plot_loss['train']['loss dsc']
                        trace1_plot = [0 for i in range(len(trace1_pre))]
                        for i in range(len(trace1_pre) - 1):
                            trace1_plot[i + 1] = (trace1_pre[i] + trace1_pre[i + 1]) / 2

                        trace2_pre = batches_plot_loss['train']['loss gen']
                        trace2_plot = [0 for i in range(len(trace2_pre))]
                        for i in range(len(trace2_pre) - 1):
                            trace2_plot[i + 1] = (trace2_pre[i] + trace2_pre[i + 1]) / 2

                        trace1 = dict(x=batch_counting['train'], y=trace1_plot, mode="markers+lines", type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"}, name='disc')
                        trace2 = dict(x=batch_counting['train'], y=trace2_plot, mode="markers+lines", type='custom', marker={'color': 'blue', 'symbol': 0, 'size': "5"}, name='gen')
                        trace3 = dict(x=batch_counting['train'], y=batches_plot_loss['train']['loss class'], mode="markers+lines", type='custom', marker={'color': 'cyan', 'symbol': 0, 'size': "5"}, name='class')
                        trace4 = dict(x=batch_counting['train'], y=batches_plot_loss['train']['loss mse'], mode="markers+lines", type='custom', marker={'color': 'purple', 'symbol': 0, 'size': "5"}, name='mse')
                        trace5 = dict(x=batch_counting['train'], y=batches_plot_loss['train']['loss kld'], mode="markers+lines", type='custom', marker={'color': 'black', 'symbol': 0, 'size': "5"}, name='kld')
                        layout = dict(title="Losses Plot (train)", xaxis={'title': 'batch idx'}, yaxis={'title': 'All losses'})
                        vis['plots']._send({'data': [trace1, trace2, trace3, trace4, trace5], 'layout': layout, 'win': 'All losses, training'})

                    self.optimizer.zero_grad()
                    L_total.backward(retain_graph=True)
                    self.check_gradients()
                    self.optimizer.step()
                    bar.update(batch_idx)

            for grouper_visualizations in range(1):
                epoch_plot_acc['train']['acc class'].append(correct_class*100.0/total_batch)
                trace = dict(x=epoch_counting['train'], y=epoch_plot_acc['train']['acc class'], mode="markers+lines", type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                layout = dict(title="Classification Accuracy Plot (train)", xaxis={'title': 'epoch idx'}, yaxis={'title': 'class accuracy'})
                vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Class accuracy, training (epoch)'})

                epoch_plot_acc['train']['acc dsc'].append(correct_discriminator * 100/ (2*total_batch))
                trace = dict(x=epoch_counting['train'], y=epoch_plot_acc['train']['acc dsc'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                layout = dict(title="Discriminator Accuracy Plot (train)", xaxis={'title': 'epoch idx'},yaxis={'title': 'class accuracy'})
                vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Discriminator accuracy, training (epoch)'})

                trace1 = dict(x=epoch_counting['train'], y=epoch_plot_loss['train']['loss dsc'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"}, name='disc')
                trace2 = dict(x=epoch_counting['train'], y=epoch_plot_loss['train']['loss gen'], mode="markers+lines",type='custom', marker={'color': 'blue', 'symbol': 0, 'size': "5"}, name='gen')
                trace3 = dict(x=epoch_counting['train'], y=epoch_plot_loss['train']['loss class'], mode="markers+lines",type='custom', marker={'color': 'cyan', 'symbol': 0, 'size': "5"}, name='class')
                trace4 = dict(x=epoch_counting['train'], y=epoch_plot_loss['train']['loss mse'], mode="markers+lines",type='custom', marker={'color': 'purple', 'symbol': 0, 'size': "5"}, name='mse')
                trace5 = dict(x=epoch_counting['train'], y=epoch_plot_loss['train']['loss kld'], mode="markers+lines",type='custom', marker={'color': 'black', 'symbol': 0, 'size': "5"}, name='kld')
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
            if epoch % self.scheduler['test'] == 0:
                epoch_test = int(math.floor(epoch/self.scheduler['test']))
                epoch_counting['test'].append(epoch_test)
                print("TESTING -----------------")
                self.GAN.GAN[0].inference = True
                self.GAN.GAN[0].eval()
                self.GAN.GAN[1].eval()

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
                        self.GAN.GAN[1].fake = not self.GAN.GAN[1].fake

                        in_datums = [Variable(datum["Image Pairs"][0].cuda(), volatile=True), Variable(datum["Image Pairs"][1].cuda(), volatile=True)]
                        stream_outputs, stream_recons, stream_inputs, mu_lists, std_lists, z_sample_lists, class_predictions, d_pred, d_labels = self.GAN(in_datums)

                        for s in range(len(in_datums)):
                            vis['rec'].image(torch.clamp(self.dbs['test'].ut(stream_inputs[s][0].cpu().data),0,1), opts={'caption':"Test Rec"})
                            vis['rec'].image(torch.clamp(self.dbs['test'].ut(stream_outputs[s][0].cpu().data),0,1),  opts={'caption':"Test Rec"})

                            if self.GAN.GAN[1].fake:
                                vis['class_dsc'].image(torch.clamp(self.dbs['train'].ut(stream_inputs[s][0].cpu().data), 0, 1), opts={'caption': 'Test target {}, pred {}'.format(d_labels[s][0], d_pred[s][0].max(0)[1].cpu().data.numpy()[0])})
                            else:
                                vis['class_dsc'].image(torch.clamp(self.dbs['train'].ut(stream_outputs[s][0].cpu().data), 0, 1), opts={'caption': 'Test target {}, pred {}'.format(d_labels[s][0], d_pred[s][0].max(0)[1].cpu().data.numpy()[0])})

                        for s in range(len(in_datums)):
                            vis['class'].image(torch.clamp(self.dbs['train'].ut(stream_inputs[s][0].cpu().data), 0, 1), opts={'caption': 'Test target {}/2, pred {}/2'.format(datum["Diff Labels"][0] + 1, 1 + class_predictions[0].max(0)[1].cpu().data.numpy()[0])})

                        class_targets = Variable(datum["Diff Labels"], volatile=True).cuda()

                        L_REC, L_KLD, L_class, L_Gen, L_Dsc = self.total_loss(stream_outputs, stream_inputs,mu_lists, std_lists, class_predictions,Variable(datum["Diff Labels"].cuda(), volatile=True),d_pred, d_labels)

                        if type(L_Gen) == type(1):
                            L_Gen = Variable(torch.zeros(1)).cuda()

                        epoch_plot_loss['test']['loss class'][epoch_test-self.scheduler['start']] += (np.double(L_class.cpu().data.numpy()))
                        epoch_plot_loss['test']['loss kld'][epoch_test-self.scheduler['start']] += (np.double(L_KLD.cpu().data.numpy()))
                        epoch_plot_loss['test']['loss mse'][epoch_test-self.scheduler['start']] += (np.double(L_REC.cpu().data.numpy()))
                        epoch_plot_loss['test']['loss gen'][epoch_test-self.scheduler['start']] += (np.double(L_Gen.cpu().data.numpy()))
                        epoch_plot_loss['test']['loss dsc'][epoch_test-self.scheduler['start']] += (np.double(L_Dsc.cpu().data.numpy()))

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

                # Visualizations
                for grouper_visualizations in range(1):
                    epoch_plot_acc['test']['acc class'].append(correct_class * 100.0 / total_batch)
                    trace = dict(x=epoch_counting['test'], y=epoch_plot_acc['test']['acc class'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                    layout = dict(title="Classification Accuracy Plot (test)", xaxis={'title': 'epoch idx'},yaxis={'title': 'class accuracy'})
                    vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Class accuracy, testing (epoch)'})

                    epoch_plot_acc['test']['acc dsc'].append(correct_discriminator * 100 / (2 * total_batch))
                    trace = dict(x=epoch_counting['test'], y=epoch_plot_acc['test']['acc dsc'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"})
                    layout = dict(title="Discriminator Accuracy Plot (test)", xaxis={'title': 'epoch idx'},yaxis={'title': 'class accuracy'})
                    vis['plots']._send({'data': [trace], 'layout': layout, 'win': 'Discriminator accuracy, testing (epoch)'})

                    trace1 = dict(x=epoch_counting['test'], y=epoch_plot_loss['test']['loss dsc'], mode="markers+lines",type='custom', marker={'color': 'red', 'symbol': 0, 'size': "5"}, name='disc')
                    trace2 = dict(x=epoch_counting['test'], y=epoch_plot_loss['test']['loss gen'], mode="markers+lines", type='custom', marker={'color': 'blue', 'symbol': 0, 'size': "5"}, name='gen')
                    trace3 = dict(x=epoch_counting['test'], y=epoch_plot_loss['test']['loss class'], mode="markers+lines", type='custom', marker={'color': 'cyan', 'symbol': 0, 'size': "5"}, name='class')
                    trace4 = dict(x=epoch_counting['test'], y=epoch_plot_loss['test']['loss mse'], mode="markers+lines",type='custom', marker={'color': 'purple', 'symbol': 0, 'size': "5"}, name='mse')
                    trace5 = dict(x=epoch_counting['test'], y=epoch_plot_loss['test']['loss kld'], mode="markers+lines",type='custom', marker={'color': 'black', 'symbol': 0, 'size': "5"}, name='kld')
                    layout = dict(title="Losses Plot (test)", xaxis={'title': 'epoch idx'}, yaxis={'title': 'All losses'})
                    vis['plots']._send({'data': [trace1, trace2, trace3, trace4, trace5], 'layout': layout,'win': 'All losses, testing (epoch)'})
            if epoch % self.scheduler['save'] == 0 and epoch != self.scheduler['start']:
                print("Saving model for epoch:{}".format(epoch))
                if self.save_models:
                    self.save_model(epoch=epoch)
            if epoch % self.scheduler['cluster'] == 0:
                print("Clustering a batch of train and test samples")

                for i, datum in enumerate(self.dbs['train']):
                    # obtain latent z's in inference mode
                    self.GAN.GAN[0].inference = True
                    self.GAN.GAN[0].eval()
                    in_datums = [Variable(datum["Image Pairs"][0].cuda(), volatile=True), Variable(datum["Image Pairs"][1].cuda(), volatile=True)]
                    stream_outputs, stream_recons, stream_inputs, mu_lists, std_lists, z_sample_lists, class_predictions = self.GAN.GAN[0](in_datums)

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

    def total_loss(self, recon_x, x, mu, logvar, class_pred, target, d_pred, d_labels, d_pred_r, d_labels_r):

        L_REC = 0
        L_KLD = 0
        L_class = 0
        L_Gen = 0
        L_Dsc = 0

        REC_multiplier = 1
        Gen_multiplier = 1

        batch_size = recon_x[0].size()[0]
        pss = self.proper_size[0] * self.proper_size[1] # proper size squared

        for stream_idx in range(len(recon_x)):
            stream_in = x[stream_idx]
            stream_out = recon_x[stream_idx]
            mu_cat = torch.cat(mu[stream_idx],dim=1)
            logvar_cat = torch.cat(logvar[stream_idx],dim=1)
            L_REC += REC_multiplier*F.mse_loss(stream_out.view(-1, 1 * pss),stream_in.view(-1, 1 * pss))
            L_KLD += -0.5 * torch.sum(1 + logvar_cat - mu_cat.pow(2) - logvar_cat.exp())

        L_KLD /= batch_size * pss * len(recon_x)
        L_KLD = torch.clamp(L_KLD, 0, 1000)

        criterion = nn.CrossEntropyLoss()
        L_class = criterion(class_pred, target.squeeze())

        for stream in range(len(recon_x)):
            # real targets are 1s, fake targets are 0s / from code for discriminator
            # We want the generator to get a small loss only if the predictions match real targets (1s)
            # since that means that we succeeded in fooling the discriminator (it predicted real labels from our reconstructions)
            discriminator_labels = Variable(torch.ones(batch_size).long()).cuda()
            if self.GAN.GAN[1].fake:
                L_Gen += Gen_multiplier*criterion(d_pred[stream], discriminator_labels)

            L_Dsc += criterion(d_pred[stream], Variable(d_labels[stream].cuda()))

        # Add a loss for the recombined samples
        if self.GAN.GAN[1].fake:
            discriminator_labels = Variable(torch.ones(batch_size).long()).cuda()
            L_Gen += Gen_multiplier * criterion(d_pred_r, discriminator_labels)
            L_Dsc += criterion(d_pred_r, Variable(d_labels_r.cuda()))

        return L_REC, L_KLD, L_class, L_Gen, L_Dsc

########################
